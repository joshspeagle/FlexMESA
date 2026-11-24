//! Dormand-Prince 5(4) Butcher tableau and Shampine dense-output polynomial.
//!
//! Stores the rational Dormand-Prince 5(4) stage abscissas $c_i$, stage weights
//! $a_{ij}$, embedded 5(4) error coefficients $e_i$, and Shampine (`1986`)
//! 4th-order continuous dense-output polynomial coefficients $d_i$ (`contd5` in
//! `num/private/mod_dopri5.f`), together with [`DopriResult`].

/// Dormand-Prince 5(4) stage abscissas $c_2, \dots, c_6$.
pub(crate) const C: [f64; 5] = [0.2, 0.3, 0.8, 8.0 / 9.0, 1.0];

/// Stage 2 coupling coefficient $a_{21}$.
pub(crate) const A21: f64 = 0.2;

/// Stage 3 coupling coefficients $[a_{31}, a_{32}]$.
pub(crate) const A3: [f64; 2] = [3.0 / 40.0, 9.0 / 40.0];

/// Stage 4 coupling coefficients $[a_{41}, a_{42}, a_{43}]$.
pub(crate) const A4: [f64; 3] = [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];

/// Stage 5 coupling coefficients $[a_{51}, a_{52}, a_{53}, a_{54}]$.
pub(crate) const A5: [f64; 4] = [
    19_372.0 / 6_561.0,
    -25_360.0 / 2_187.0,
    64_448.0 / 6_561.0,
    -212.0 / 729.0,
];

/// Stage 6 coupling coefficients $[a_{61}, \dots, a_{65}]$.
pub(crate) const A6: [f64; 5] = [
    9_017.0 / 3_168.0,
    -355.0 / 33.0,
    46_732.0 / 5_247.0,
    49.0 / 176.0,
    -5_103.0 / 18_656.0,
];

/// Stage 7 (FSAL 5th-order solution) weights $[a_{71}, \dots, a_{76}]$.
pub(crate) const A7: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
];

/// Embedded 5(4) local error weights $[e_1, \dots, e_7]$.
pub(crate) const E: [f64; 7] = [
    71.0 / 57_600.0,
    0.0,
    -71.0 / 16_695.0,
    71.0 / 1_920.0,
    -17_253.0 / 339_200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

/// Shampine 4th-order dense-output coefficients $[d_1, \dots, d_7]$.
pub(crate) const D: [f64; 7] = [
    -12_715_105_075.0 / 11_282_082_432.0,
    0.0,
    87_487_479_700.0 / 32_700_410_799.0,
    -10_690_763_975.0 / 1_880_347_072.0,
    701_980_252_875.0 / 199_316_789_632.0,
    -1_453_857_185.0 / 822_651_844.0,
    69_997_945.0 / 29_380_423.0,
];

/// Result of an adaptive Dormand-Prince (`dopri5` / `dop853`) integration.
#[derive(Debug, Clone)]
pub struct DopriResult {
    /// Final state vector $y(x_{\mathrm{end}})$ of length $n$.
    pub y: Vec<f64>,
    /// Total number of right-hand side evaluations (`nfcn`).
    pub nfcn: usize,
    /// Total number of steps attempted (`nstep`).
    pub nstep: usize,
    /// Number of accepted steps (`naccpt`).
    pub naccpt: usize,
    /// Number of rejected steps (`nrejct`).
    pub nrejct: usize,
    /// Dense-output polynomial history `(x_old, h, rcont[5][n])` per step.
    pub history: Vec<(f64, f64, [Vec<f64>; 5])>,
}

impl DopriResult {
    /// Evaluates Shampine's 4th-order dense output polynomial (`contd5`) at any
    /// abscissa `x_query` within the integration interval using normalized
    /// step fraction $\theta = (x - x_{\mathrm{old}}) / h$ and nested Horner
    /// evaluation over $r_1, \dots, r_5$.
    #[must_use]
    pub fn dense_eval(&self, x_query: f64) -> Vec<f64> {
        if self.history.is_empty() {
            return self.y.clone();
        }
        let mut chosen = &self.history[self.history.len() - 1];
        for seg in &self.history {
            let (x0, h, _) = *seg;
            let xmin = x0.min(x0 + h);
            let xmax = x0.max(x0 + h);
            if x_query >= xmin - 1e-14 && x_query <= xmax + 1e-14 {
                chosen = seg;
                break;
            }
        }
        let (x0, h, ref rc) = *chosen;
        let theta = if h == 0.0 { 0.0 } else { (x_query - x0) / h };
        let theta1 = 1.0 - theta;
        let n = rc[0].len();
        (0..n)
            .map(|i| {
                rc[0][i]
                    + theta
                        * (rc[1][i]
                            + theta1
                                * (rc[2][i]
                                    + theta * (rc[3][i] + theta1 * rc[4][i])))
            })
            .collect()
    }
}

/// Hairer-Wanner DOP853 12-stage embedded 8(5,3) Runge-Kutta tableau
/// coefficients verbatim from `num/private/mod_dop853.f:223-311`.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
pub(crate) mod dop853 {
    pub(crate) const C2: f64 = 0.526001519587677318785587544488e-01;
    pub(crate) const C3: f64 = 0.789002279381515978178381316732e-01;
    pub(crate) const C4: f64 = 0.118350341907227396726757197510e+00;
    pub(crate) const C5: f64 = 0.281649658092772603273242802490e+00;
    pub(crate) const C6: f64 = 0.333333333333333333333333333333e+00;
    pub(crate) const C7: f64 = 0.25e+00;
    pub(crate) const C8: f64 = 0.307692307692307692307692307692e+00;
    pub(crate) const C9: f64 = 0.651282051282051282051282051282e+00;
    pub(crate) const C10: f64 = 0.6e+00;
    pub(crate) const C11: f64 = 0.857142857142857142857142857142e+00;

    pub(crate) const B1: f64 = 5.42937341165687622380535766363e-2;
    pub(crate) const B6: f64 = 4.45031289275240888144113950566;
    pub(crate) const B7: f64 = 1.89151789931450038304281599044;
    pub(crate) const B8: f64 = -5.8012039600105847814672114227;
    pub(crate) const B9: f64 = 3.1116436695781989440891606237e-1;
    pub(crate) const B10: f64 = -1.52160949662516078556178806805e-1;
    pub(crate) const B11: f64 = 2.01365400804030348374776537501e-1;
    pub(crate) const B12: f64 = 4.47106157277725905176885569043e-2;

    pub(crate) const BHH1: f64 = 0.244094488188976377952755905512;
    pub(crate) const BHH2: f64 = 0.733846688281611857341361741547;
    pub(crate) const BHH3: f64 = 0.220588235294117647058823529412e-01;

    pub(crate) const ER1: f64 = 0.1312004499419488073250102996e-01;
    pub(crate) const ER6: f64 = -0.1225156446376204440720569753e+01;
    pub(crate) const ER7: f64 = -0.4957589496572501915214079952;
    pub(crate) const ER8: f64 = 0.1664377182454986536961530415e+01;
    pub(crate) const ER9: f64 = -0.3503288487499736816886487290;
    pub(crate) const ER10: f64 = 0.3341791187130174790297318841;
    pub(crate) const ER11: f64 = 0.8192320648511571246570742613e-01;
    pub(crate) const ER12: f64 = -0.2235530786388629525884427845e-01;

    pub(crate) const A21: f64 = 5.26001519587677318785587544488e-2;
    pub(crate) const A31: f64 = 1.97250569845378994544595329183e-2;
    pub(crate) const A32: f64 = 5.91751709536136983633785987549e-2;
    pub(crate) const A41: f64 = 2.95875854768068491816892993775e-2;
    pub(crate) const A43: f64 = 8.87627564304205475450678981324e-2;
    pub(crate) const A51: f64 = 2.41365134159266685502369798665e-1;
    pub(crate) const A53: f64 = -8.84549479328286085344864962717e-1;
    pub(crate) const A54: f64 = 9.24834003261792003115737966543e-1;
    pub(crate) const A61: f64 = 3.7037037037037037037037037037e-2;
    pub(crate) const A64: f64 = 1.70828608729473871279604482173e-1;
    pub(crate) const A65: f64 = 1.25467687566822425016691814123e-1;
    pub(crate) const A71: f64 = 3.7109375e-2;
    pub(crate) const A74: f64 = 1.70252211019544039314978060272e-1;
    pub(crate) const A75: f64 = 6.02165389804559606850219397283e-2;
    pub(crate) const A76: f64 = -1.7578125e-2;
    pub(crate) const A81: f64 = 3.70920001185047927108779319836e-2;
    pub(crate) const A84: f64 = 1.70383925712239993810214054705e-1;
    pub(crate) const A85: f64 = 1.07262030446373284651809199168e-1;
    pub(crate) const A86: f64 = -1.53194377486244017527936158236e-2;
    pub(crate) const A87: f64 = 8.27378916381402288758473766002e-3;
    pub(crate) const A91: f64 = 6.24110958716075717114429577812e-1;
    pub(crate) const A94: f64 = -3.36089262944694129406857109825;
    pub(crate) const A95: f64 = -8.68219346841726006818189891453e-1;
    pub(crate) const A96: f64 = 2.75920996994467083049415600797e1;
    pub(crate) const A97: f64 = 2.01540675504778934086186788979e1;
    pub(crate) const A98: f64 = -4.34898841810699588477366255144e1;
    pub(crate) const A101: f64 = 4.77662536438264365890433908527e-1;
    pub(crate) const A104: f64 = -2.48811461997166764192642586468;
    pub(crate) const A105: f64 = -5.90290826836842996371446475743e-1;
    pub(crate) const A106: f64 = 2.12300514481811942347288949897e1;
    pub(crate) const A107: f64 = 1.52792336328824235832596922938e1;
    pub(crate) const A108: f64 = -3.32882109689848629194453265587e1;
    pub(crate) const A109: f64 = -2.03312017085086261358222928593e-2;
    pub(crate) const A111: f64 = -9.3714243008598732571704021658e-1;
    pub(crate) const A114: f64 = 5.18637242884406370830023853209;
    pub(crate) const A115: f64 = 1.09143734899672957818500254654;
    pub(crate) const A116: f64 = -8.14978701074692612513997267357;
    pub(crate) const A117: f64 = -1.85200656599969598641566180701e1;
    pub(crate) const A118: f64 = 2.27394870993505042818970056734e1;
    pub(crate) const A119: f64 = 2.49360555267965238987089396762;
    pub(crate) const A1110: f64 = -3.0467644718982195003823669022;
    pub(crate) const A121: f64 = 2.27331014751653820792359768449;
    pub(crate) const A124: f64 = -1.05344954667372501984066689879e1;
    pub(crate) const A125: f64 = -2.00087205822486249909675718444;
    pub(crate) const A126: f64 = -1.79589318631187989172765950534e1;
    pub(crate) const A127: f64 = 2.79488845294199600508499808837e1;
    pub(crate) const A128: f64 = -2.85899827713502369474065508674;
    pub(crate) const A129: f64 = -8.87285693353062954433549289258;
    pub(crate) const A1210: f64 = 1.23605671757943030647266201528e1;
    pub(crate) const A1211: f64 = 6.43392746015763530355970484046e-1;
}
