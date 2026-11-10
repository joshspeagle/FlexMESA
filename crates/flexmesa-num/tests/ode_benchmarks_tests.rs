//! Golden verification of `flexmesa-num` ODE/DAE/PDE solvers against upstream
//! MESA `num/test/test_output` (`vdpol`, `medakzo`, `chemakzo`, `beam`,
//! `pollu`, and `test_diffusion`).

#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::manual_memcpy,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unnecessary_wraps
)]

use flexmesa_num::{
    dop853, dopri5, rosenbrock_banded, rosenbrock_dae_diag, rosenbrock_dense,
    NumError, RosenbrockVariant,
};

const ALL_ROS: [RosenbrockVariant; 7] = [
    RosenbrockVariant::Ros2,
    RosenbrockVariant::Rose2,
    RosenbrockVariant::Ros3p,
    RosenbrockVariant::Ros3pl,
    RosenbrockVariant::Rodas3,
    RosenbrockVariant::Rodas4,
    RosenbrockVariant::Rodasp,
];

/// Verifies `vdpol` ($n = 2, \mu = 10^{-3}, t \in [0, 2.0]$) across all 9
/// solvers (`dopri5`, `dop853`, and all 7 Rosenbrock solvers) against
/// `num/test/test_output:243-386`.
#[test]
fn test_vdpol_all_nine_solvers_match_golden_test_output() {
    let mu = 1e-3_f64;
    let y0 = [2.0_f64, 0.0_f64];
    let y_ref = [1.763_234_540_188_910_2, -0.835_688_681_914_662_1];
    let fcn = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        Ok(vec![y[1], ((1.0 - y[0] * y[0]) * y[1] - y[0]) / mu])
    };
    let jac = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        Ok(vec![
            0.0,
            1.0,
            (-2.0 * y[0] * y[1] - 1.0) / mu,
            (1.0 - y[0] * y[0]) / mu,
        ])
    };

    let r5 = dopri5(0.0, 2.0, &y0, 1e-6, 0.0, 1e-4, 1e-4, 100_000, fcn).unwrap();
    let r8 = dop853(0.0, 2.0, &y0, 1e-6, 0.0, 1e-4, 1e-4, 100_000, fcn).unwrap();
    for i in 0..2 {
        assert!((r5.y[i] - y_ref[i]).abs() <= 2e-3);
        assert!((r8.y[i] - y_ref[i]).abs() <= 2e-3);
    }

    for var in ALL_ROS {
        let res = rosenbrock_dense(
            var, 0.0, 2.0, &y0, 1e-4, 0.0, 1e-4, 1e-4, 100_000, fcn, jac,
        )
        .unwrap();
        for i in 0..2 {
            assert!(
                (res.y[i] - y_ref[i]).abs() <= 2.5e-3,
                "vdpol {var:?} comp {i}: got {}, ref {}",
                res.y[i],
                y_ref[i]
            );
        }
    }
}

/// Evaluates `medakzo` ($N = 200, n = 400$) RHS (`bari_medakzo.f90:40-81`).
fn medakzo_feval(t: f64, y: &[f64]) -> Result<Vec<f64>, NumError> {
    let n = y.len();
    let n_pts = n / 2;
    let dz = 1.0 / (n_pts as f64);
    let dz2 = dz * dz;
    let k_rate = 100.0_f64;
    let c = 4.0_f64;
    let phi = if t <= 5.0 { 2.0 } else { 0.0 };

    let mut f = vec![0.0_f64; n];
    let dum1 = (dz - 1.0) * (dz - 1.0) / c;
    let alpha1 = 2.0 * (dz - 1.0) * dum1 / c;
    let beta1 = dum1 * dum1;
    f[0] = (phi - 2.0 * y[0] + y[2]) * beta1 / dz2
        + alpha1 * (y[2] - phi) / (2.0 * dz)
        - k_rate * y[0] * y[1];
    f[1] = -k_rate * y[0] * y[1];

    for j in 2..n_pts {
        let i = 2 * j - 2;
        let zeta = (j as f64) * dz;
        let dum = (zeta - 1.0) * (zeta - 1.0) / c;
        let alpha = 2.0 * (zeta - 1.0) * dum / c;
        let beta = dum * dum;
        let gama = (y[i - 2] - 2.0 * y[i] + y[i + 2]) * beta / dz2
            + alpha * (y[i + 2] - y[i - 2]) / (2.0 * dz);
        f[i] = gama - k_rate * y[i] * y[i + 1];
        f[i + 1] = -k_rate * y[i + 1] * y[i];
    }
    f[n - 2] = -k_rate * y[n - 2] * y[n - 1];
    f[n - 1] = -k_rate * y[n - 2] * y[n - 1];
    Ok(f)
}

/// Evaluates `medakzo` dense $400 \times 400$ Jacobian (`m_l = 2, m_u = 2`).
fn medakzo_jeval(_t: f64, y: &[f64]) -> Result<Vec<f64>, NumError> {
    let n = y.len();
    let n_pts = n / 2;
    let dz = 1.0 / (n_pts as f64);
    let dz2 = dz * dz;
    let k_rate = 100.0_f64;
    let c = 4.0_f64;

    let mut jac = vec![0.0_f64; n * n];
    let dum1 = (dz - 1.0) * (dz - 1.0) / c;
    let alpha1 = 2.0 * (dz - 1.0) * dum1 / c;
    let beta1 = dum1 * dum1;
    jac[0] = -2.0 * beta1 / dz2 - k_rate * y[1];
    jac[1] = -k_rate * y[0];
    jac[2] = beta1 / dz2 + alpha1 / (2.0 * dz);
    jac[n] = -k_rate * y[1];
    jac[n + 1] = -k_rate * y[0];

    for j in 2..n_pts {
        let i = 2 * j - 2;
        let zeta = (j as f64) * dz;
        let dum = (zeta - 1.0) * (zeta - 1.0) / c;
        let alpha = 2.0 * (zeta - 1.0) * dum / c;
        let beta = dum * dum;
        let bz = beta / dz2;
        jac[i * n + (i - 2)] = bz - alpha / (2.0 * dz);
        jac[i * n + i] = -2.0 * bz - k_rate * y[i + 1];
        jac[i * n + (i + 1)] = -k_rate * y[i];
        jac[i * n + (i + 2)] = bz + alpha / (2.0 * dz);
        jac[(i + 1) * n + i] = -k_rate * y[i + 1];
        jac[(i + 1) * n + (i + 1)] = -k_rate * y[i];
    }
    jac[(n - 2) * n + (n - 2)] = -k_rate * y[n - 1];
    jac[(n - 2) * n + (n - 1)] = -k_rate * y[n - 2];
    jac[(n - 1) * n + (n - 2)] = -k_rate * y[n - 1];
    jac[(n - 1) * n + (n - 1)] = -k_rate * y[n - 2];
    Ok(jac)
}

/// Verifies `medakzo` ($n = 400$, banded $m_l = 2, m_u = 2$, $t \in [0, 5] \cup
/// [5, 20]$) across `Ros3pl`, `Rodas3`, `Rodas4`, and `Rodasp` against
/// `num/test/test_output:480-583`.
#[test]
fn test_medakzo_banded_matches_golden_test_output() {
    let n = 400usize;
    let mut y0 = vec![0.0_f64; n];
    for j in 0..(n / 2) {
        y0[2 * j + 1] = 1.0;
    }
    let ref_11 = [
        5.113_983_840_919_909e-6,
        1.052_886_195_582_22e-4,
        2.156_525_324_601_176e-4,
        3.169_160_480_759_169e-4,
        3.616_265_691_144_918e-4,
        2.582_807_380_350_103e-4,
        7.048_158_429_518_009e-7,
        8.721_783_456_154_47e-16,
        1.739_533_019_752_215e-29,
        1.247_057_764_661_705e-51,
        6.344_600_529_877_694e-89,
    ];

    for var in [
        RosenbrockVariant::Ros3pl,
        RosenbrockVariant::Rodas3,
        RosenbrockVariant::Rodas4,
        RosenbrockVariant::Rodasp,
    ] {
        let r1 = rosenbrock_banded(
            var, 2, 2, 0.0, 5.0, &y0, 1e-9, 0.0, 1e-6, 1e-6, 50_000,
            medakzo_feval, medakzo_jeval,
        )
        .unwrap();
        let r2 = rosenbrock_banded(
            var, 2, 2, 5.0, 20.0, &r1.y, 1e-9, 0.0, 1e-6, 1e-6, 50_000,
            medakzo_feval, medakzo_jeval,
        )
        .unwrap();
        for (idx, &expected) in ref_11.iter().enumerate() {
            let k = 1 + 18 * idx; // 1, 19, 37, ..., 181
            let comp = 2 * (k - 1);
            assert!(
                (r2.y[comp] - expected).abs() <= 5e-5,
                "medakzo {var:?} idx {idx}: got {}, ref {expected}",
                r2.y[comp]
            );
        }
    }
}

/// Verifies `chemakzo` ($n = 6$ index-1 DAE with diagonal mass matrix
/// $M = \mathrm{diag}(1, 1, 1, 1, 1, 0)$ on $t \in [0, 180]$) against
/// `bari_chemakzo.f90:211-216`.
#[test]
fn test_chemakzo_dae_matches_golden_reference() {
    let ks = 115.83_f64;
    let k1 = 18.7_f64;
    let k2 = 0.58_f64;
    let k3 = 0.09_f64;
    let k4 = 0.42_f64;
    let kbig = 34.4_f64;
    let kla = 3.3_f64;
    let po2 = 0.9_f64;
    let hen = 737.0_f64;

    let y0 = [0.444, 0.00123, 0.0, 0.007, 0.0, ks * 0.444 * 0.007];
    let mass = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let y_ref = [
        0.115_079_492_066_170_2,
        0.120_383_147_156_771_5e-2,
        0.161_156_288_740_797_4,
        0.365_615_642_124_928_3e-3,
        0.170_801_088_526_440_4e-1,
        0.487_353_131_030_745_5e-2,
    ];

    let fcn = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        let sqy2 = y[1].max(0.0).sqrt();
        let r1 = k1 * y[0].powi(4) * sqy2;
        let r2 = k2 * y[2] * y[3];
        let r3 = (k2 / kbig) * y[0] * y[4];
        let r4 = k3 * y[0] * y[3] * y[3];
        let r5 = k4 * y[5] * y[5] * sqy2;
        let fin = kla * (po2 / hen - y[1]);
        Ok(vec![
            -2.0 * r1 + r2 - r3 - r4,
            -0.5 * r1 - r4 - 0.5 * r5 + fin,
            r1 - r2 + r3,
            -r2 + r3 - 2.0 * r4,
            r2 - r3 + r5,
            ks * y[0] * y[3] - y[5],
        ])
    };
    let jac = |t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        let n = 6;
        let f0 = fcn(t, y)?;
        let mut j = vec![0.0; n * n];
        for col in 0..n {
            let mut yp = y.to_vec();
            let eps = 1e-8 * yp[col].abs().max(1e-6);
            yp[col] += eps;
            let fp = fcn(t, &yp)?;
            for row in 0..n {
                j[row * n + col] = (fp[row] - f0[row]) / eps;
            }
        }
        Ok(j)
    };

    let res = rosenbrock_dae_diag(
        RosenbrockVariant::Rodas4,
        &mass,
        0.0,
        180.0,
        &y0,
        1e-10,
        0.0,
        1e-7,
        1e-7,
        100_000,
        fcn,
        jac,
    )
    .unwrap();
    for i in 0..6 {
        assert!((res.y[i] - y_ref[i]).abs() <= 1e-5);
    }
}

/// Evaluates `beam` ($N = 40, n = 80$) RHS (`bari_beam.f90:41-130`).
fn beam_feval(t: f64, th: &[f64]) -> Result<Vec<f64>, NumError> {
    let n = 40usize;
    let nn = 80usize;
    let nsq = (n * n) as f64;
    let nquatr = nsq * nsq;
    let mut sth = vec![0.0_f64; n + 1];
    let mut cth = vec![0.0_f64; n + 1];
    for i in 2..=n {
        let d = th[i - 1] - th[i - 2];
        sth[i] = d.sin();
        cth[i] = d.cos();
    }
    let mut v = vec![0.0_f64; n + 1];
    let (fx, fy) = if t > std::f64::consts::PI {
        (0.0, 0.0)
    } else {
        let fabs = 1.5 * t.sin().powi(2);
        (-fabs, fabs)
    };
    v[1] = (-3.0 * th[0] + th[1]) * nquatr
        + nsq * (fy * th[0].cos() - fx * th[0].sin());
    for i in 2..n {
        v[i] = (th[i - 2] - 2.0 * th[i - 1] + th[i]) * nquatr
            + nsq * (fy * th[i - 1].cos() - fx * th[i - 1].sin());
    }
    v[n] = (th[n - 2] - th[n - 1]) * nquatr
        + nsq * (fy * th[n - 1].cos() - fx * th[n - 1].sin());

    let mut w = vec![0.0_f64; n + 1];
    w[1] = sth[2] * v[2];
    for i in 2..n {
        w[i] = -sth[i] * v[i - 1] + sth[i + 1] * v[i + 1];
    }
    w[n] = -sth[n] * v[n - 1];
    for i in 1..=n {
        w[i] += th[n + i - 1].powi(2);
    }

    let mut alpha = vec![2.0_f64; n + 1];
    let mut beta = vec![0.0_f64; n + 1];
    alpha[1] = 1.0;
    for i in 2..=n {
        beta[i - 1] = -cth[i];
    }
    alpha[n] = 3.0;
    for i in (1..n).rev() {
        let q = beta[i] / alpha[i + 1];
        w[i] -= w[i + 1] * q;
        alpha[i] -= beta[i] * q;
    }
    w[1] /= alpha[1];
    for i in 2..=n {
        w[i] = (w[i] - beta[i - 1] * w[i - 1]) / alpha[i];
    }

    let mut u = vec![0.0_f64; n + 1];
    u[1] = v[1] - cth[2] * v[2] + sth[2] * w[2];
    for i in 2..n {
        u[i] = 2.0 * v[i]
            - cth[i] * v[i - 1]
            - cth[i + 1] * v[i + 1]
            - sth[i] * w[i - 1]
            + sth[i + 1] * w[i + 1];
    }
    u[n] = 3.0 * v[n] - cth[n] * v[n - 1] - sth[n] * w[n - 1];

    let mut df = vec![0.0_f64; nn];
    for i in 0..n {
        df[i] = th[n + i];
        df[n + i] = u[i + 1];
    }
    Ok(df)
}

/// Verifies `beam` ($n = 80, t \in [0, 5]$) against `bari_beam.f90:151-230`
/// and `pollu` ($n = 20, t \in [0, 60]$) against `test_pollu.f90`.
#[test]
fn test_beam_and_pollu_match_golden_references() {
    // 1. beam (n = 80) on [0, pi] then [pi, 5] via dop853
    let y0_beam = vec![0.0_f64; 80];
    let r_pi = dop853(
        0.0,
        std::f64::consts::PI,
        &y0_beam,
        1e-4,
        0.0,
        1e-6,
        1e-6,
        200_000,
        beam_feval,
    )
    .unwrap();
    let r_beam = dop853(
        std::f64::consts::PI,
        5.0,
        &r_pi.y,
        1e-4,
        0.0,
        1e-6,
        1e-6,
        200_000,
        beam_feval,
    )
    .unwrap();
    let ind_beam = [0usize, 9, 19, 29, 39, 49, 59, 69, 79];
    let ref_beam = [
        -0.579_236_659_128_500_7e-2,
        -0.911_913_465_464_794_7e-1,
        -0.148_359_547_246_301_2,
        -0.172_582_184_746_253_7,
        -0.176_720_176_107_548_8,
        0.598_760_970_262_427,
        0.986_874_782_172_836_3,
        1.156_792_132_004_482,
        1.186_724_615_113_034,
    ];
    for (idx, &comp) in ind_beam.iter().enumerate() {
        assert!((r_beam.y[comp] - ref_beam[idx]).abs() <= 5e-3);
    }

    // 2. pollu (n = 20) on [0, 60] via Rodas4
    let mut y0_pollu = [0.0_f64; 20];
    y0_pollu[1] = 0.2;
    y0_pollu[3] = 0.04;
    y0_pollu[6] = 0.1;
    y0_pollu[7] = 0.3;
    y0_pollu[8] = 0.01;
    y0_pollu[16] = 0.007;
    let pollu_f = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        let r = [
            0.0,
            0.35 * y[0],
            26.6 * y[1] * y[3],
            1.23e4 * y[4] * y[1],
            8.6e-4 * y[6],
            8.2e-4 * y[6],
            1.5e4 * y[6] * y[5],
            1.3e-4 * y[8],
            2.4e4 * y[8] * y[5],
            1.65e4 * y[10] * y[1],
            9.0e3 * y[10] * y[0],
            2.2e-2 * y[12],
            1.2e4 * y[9] * y[1],
            1.88 * y[13],
            1.63e4 * y[0] * y[5],
            4.8e6 * y[2],
            3.5e-4 * y[3],
            1.75e-2 * y[3],
            1.0e8 * y[15],
            4.44e11 * y[15],
            1.24e3 * y[16] * y[5],
            2.1 * y[18],
            5.78 * y[18],
            4.74e-2 * y[0] * y[3],
            1.78e3 * y[18] * y[0],
            3.12 * y[19],
        ];
        Ok(vec![
            -r[1] - r[10] - r[14] - r[23] - r[24]
                + r[2] + r[3] + r[9] + r[11] + r[12] + r[22] + r[25],
            -r[2] - r[3] - r[9] - r[12] + r[1] + r[21],
            -r[15] + r[1] + r[17] + r[19] + r[22],
            -r[2] - r[16] - r[17] - r[23] + r[15],
            -r[3] + 2.0 * r[4] + r[6] + r[7] + r[13] + r[20],
            -r[6] - r[8] - r[14] - r[20] + r[3] + 2.0 * r[18],
            -r[4] - r[5] - r[6] + r[13],
            r[4] + r[5] + r[6] + r[7],
            -r[7] - r[8],
            -r[12] + r[7] + r[9],
            -r[9] - r[10] + r[8] + r[11],
            r[9],
            -r[11] + r[10],
            -r[13] + r[12],
            r[14],
            -r[18] - r[19] + r[16],
            -r[20],
            r[20],
            -r[21] - r[22] - r[24] + r[23] + r[25],
            -r[25] + r[24],
        ])
    };
    let pollu_j = |t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        let f0 = pollu_f(t, y)?;
        let mut j = vec![0.0; 400];
        for c in 0..20 {
            let mut yp = y.to_vec();
            let eps = 1e-8 * yp[c].abs().max(1e-8);
            yp[c] += eps;
            let fp = pollu_f(t, &yp)?;
            for r in 0..20 {
                j[r * 20 + c] = (fp[r] - f0[r]) / eps;
            }
        }
        Ok(j)
    };
    let r_pollu = rosenbrock_dense(
        RosenbrockVariant::Rodas4,
        0.0,
        60.0,
        &y0_pollu,
        1e-7,
        0.0,
        1e-5,
        1e-5,
        50_000,
        pollu_f,
        pollu_j,
    )
    .unwrap();
    let ref_pollu = [
        0.564_625_548_002_276_9e-1,
        0.134_248_413_042_233_9,
        0.413_973_433_109_942_7e-8,
        0.552_314_020_748_435_9e-2,
        0.201_897_726_230_219_6e-6,
        0.146_454_186_349_396_6e-6,
        0.778_424_911_899_796_4e-1,
        0.324_507_535_339_601_8,
        0.749_401_338_388_040_6e-2,
        0.162_229_315_730_156_1e-7,
        0.113_586_383_325_707_5e-7,
        0.223_050_597_572_135_9e-2,
        0.208_716_288_279_863e-3,
        0.139_692_101_684_015_8e-4,
        0.896_488_485_689_829_5e-2,
        0.435_284_636_933_010_3e-17,
        0.689_921_969_626_340_5e-2,
        0.100_780_303_736_594_6e-3,
        0.177_214_651_396_998_4e-5,
        0.568_294_329_231_639_2e-4,
    ];
    for i in 0..20 {
        assert!((r_pollu.y[i] - ref_pollu[i]).abs() <= 5e-5);
    }
}

/// Verifies `test_diffusion` ($n = 48, m_l = 1, m_u = 1, t \in [0, 10^4]$)
/// across all 7 Rosenbrock solvers against `num/test/test_output:585-752`.
#[test]
fn test_diffusion_all_seven_solvers_match_golden_test_output() {
    let nz = 48usize;
    let mut y0 = vec![0.0_f64; nz];
    for item in y0.iter_mut().take(nz).skip(nz / 2) {
        *item = 1.0;
    }
    let mut sig_dm = vec![0.0_f64; nz];
    for k in 1..=nz {
        if k > 1 && 5 * k > nz && 5 * k < 4 * nz {
            sig_dm[k - 1] = 1e-2;
        }
    }
    let fcn = |_t: f64, y: &[f64]| -> Result<Vec<f64>, NumError> {
        let mut f = vec![0.0_f64; nz];
        let mut sig2 = 0.0_f64;
        for i in 0..nz {
            let sig1 = sig2;
            sig2 = if i + 1 < nz { sig_dm[i + 1] } else { 0.0 };
            f[i] = if i + 1 == nz {
                sig1 * (y[i - 1] - y[i])
            } else if i == 0 {
                -sig2 * (y[0] - y[1])
            } else {
                sig1 * (y[i - 1] - y[i]) - sig2 * (y[i] - y[i + 1])
            };
        }
        Ok(f)
    };
    let jac = |_t: f64, _y: &[f64]| -> Result<Vec<f64>, NumError> {
        let mut j = vec![0.0_f64; nz * nz];
        let mut sig2 = 0.0_f64;
        for i in 0..nz {
            let sig1 = sig2;
            sig2 = if i + 1 < nz { sig_dm[i + 1] } else { 0.0 };
            if i > 0 {
                j[i * nz + (i - 1)] = sig1;
            }
            j[i * nz + i] = -(sig1 + sig2);
            if i + 1 < nz {
                j[i * nz + (i + 1)] = sig2;
            }
        }
        Ok(j)
    };
    let ref_10 = [
        2.583_083_225_876_528e-1,
        2.784_700_206_379_522e-1,
        3.169_124_322_991_038e-1,
        3.700_196_680_615_955e-1,
        4.327_355_331_324_393e-1,
        4.990_030_779_485_014e-1,
        5.623_280_759_836_704e-1,
        6.164_209_050_718_791e-1,
        6.558_489_044_788_533e-1,
        6.766_197_264_650_193e-1,
    ];
    for var in ALL_ROS {
        let res = rosenbrock_banded(
            var, 1, 1, 0.0, 1e4, &y0, 1e-7, 0.0, 1e-6, 1e-6, 50_000, fcn, jac,
        )
        .unwrap();
        for (idx, &expected) in ref_10.iter().enumerate() {
            let k = 10 + 3 * idx; // 10, 13, ..., 37 (1-based)
            let tol = if var == RosenbrockVariant::Ros3p {
                5e-3
            } else {
                1e-5
            };
            assert!(
                (res.y[k - 1] - expected).abs() <= tol,
                "diffusion {var:?} idx {idx}: got {}, ref {expected}",
                res.y[k - 1]
            );
        }
    }
}
