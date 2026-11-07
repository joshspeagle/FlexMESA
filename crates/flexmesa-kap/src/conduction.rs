//! Electron thermal conduction opacity (`condint.f90`).
//!
//! Implements the Potekhin (1999) / Cassisi et al. (2007) electron thermal
//! conductivity bicubic spline table (`condtabl.data`) and the Blouin et al.
//! (2020, ApJ 899, 46) analytical moderate-coupling correction for H and He.

use crate::spline::do_kap_interpolations;
use crate::types::{KapConfig, KapError};
use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::{BOLTZ_SIGMA, ILN10};
use flexmesa_interp2d::{interp_mkbicub_db, BicubicBc};
use flexmesa_math::exp10;
use std::sync::OnceLock;

const NUM_LOG_RHOS: usize = 71;
const NUM_LOG_TS: usize = 29;
const NUM_LOG_ZS: usize = 15;

const CONDTABL_STR: &str = include_str!("../data/condtabl.data");

/// Parsed and spline-fitted Potekhin electron thermal conductivity table (`condtabl.data`).
pub struct PotekhinTable {
    /// Tabulated $\log_{10}(\rho / \mathrm{g\,cm^{-3}})$ grid (`71` points, `[-6.0, 11.5]`).
    pub log_rhos: [f64; NUM_LOG_RHOS],
    /// Tabulated $\log_{10}(T / \mathrm{K})$ grid (`29` points, `[3.0, 10.0]`).
    pub log_ts: [f64; NUM_LOG_TS],
    /// Tabulated $\log_{10} \bar{Z}$ grid (`15` points, $Z \in [1, 60]$).
    pub log_zs: [f64; NUM_LOG_ZS],
    /// Per-$Z$ bicubic spline coefficients (`15` slices of `71 * 29` `[f, f_xx, f_yy, f_xxyy]`).
    pub f_ary: Vec<Vec<[f64; 4]>>,
}

static POTEKHIN: OnceLock<PotekhinTable> = OnceLock::new();

/// Return a reference to the lazily initialized Potekhin electron conduction table.
#[must_use]
pub fn potekhin_table() -> &'static PotekhinTable {
    POTEKHIN.get_or_init(init_potekhin)
}

fn init_potekhin() -> PotekhinTable {
    let mut lines = CONDTABL_STR.lines();
    let _header = lines.next();

    let mut log_rhos = [0.0_f64; NUM_LOG_RHOS];
    let mut log_ts = [0.0_f64; NUM_LOG_TS];
    let mut log_zs = [0.0_f64; NUM_LOG_ZS];
    let mut f_ary = Vec::with_capacity(NUM_LOG_ZS);

    for log_z_slot in &mut log_zs {
        let z_line = lines.next().expect("missing Z line in condtabl.data");
        let mut z_toks = z_line.split_whitespace();
        let z_val: f64 = z_toks
            .next()
            .expect("empty Z line")
            .parse()
            .expect("invalid Z float");
        *log_z_slot = if (z_val - 1.0).abs() == 0.0 {
            0.0
        } else {
            z_val.log10()
        };
        for log_t_slot in &mut log_ts {
            *log_t_slot = z_toks
                .next()
                .expect("missing logT in condtabl.data")
                .parse()
                .expect("invalid logT float");
        }

        let mut z_slice = vec![0.0_f64; NUM_LOG_RHOS * NUM_LOG_TS];
        for i_r in 0..NUM_LOG_RHOS {
            let r_line =
                lines.next().expect("missing logRho row in condtabl.data");
            let mut r_toks = r_line.split_whitespace();
            log_rhos[i_r] = r_toks
                .next()
                .expect("missing logRho in condtabl.data")
                .parse()
                .expect("invalid logRho float");
            for j_t in 0..NUM_LOG_TS {
                z_slice[i_r * NUM_LOG_TS + j_t] = r_toks
                    .next()
                    .expect("missing cond value in condtabl.data")
                    .parse()
                    .expect("invalid cond float");
            }
        }

        let (coeffs, _, _) = interp_mkbicub_db(
            &log_rhos,
            &log_ts,
            &z_slice,
            BicubicBc::ZeroFirstDeriv,
            &[],
            BicubicBc::ZeroFirstDeriv,
            &[],
            BicubicBc::ZeroFirstDeriv,
            &[],
            BicubicBc::ZeroFirstDeriv,
            &[],
        )
        .expect("interp_mkbicub_db failed on condtabl.data");
        f_ary.push(coeffs);
    }

    PotekhinTable {
        log_rhos,
        log_ts,
        log_zs,
        f_ary,
    }
}

fn eval_potekhin_z_slice(
    tbl: &PotekhinTable,
    iz: usize,
    log_rho: f64,
    log_t: f64,
) -> (f64, f64, f64) {
    let f1 = &tbl.f_ary[iz];
    let i_log_rho = if log_rho < tbl.log_rhos[1] {
        0
    } else {
        let mut found = NUM_LOG_RHOS - 2;
        for k in 1..(NUM_LOG_RHOS - 1) {
            if log_rho >= tbl.log_rhos[k] && log_rho < tbl.log_rhos[k + 1] {
                found = k;
                break;
            }
        }
        found
    };
    let log_rho0 = tbl.log_rhos[i_log_rho];
    let log_rho1 = tbl.log_rhos[i_log_rho + 1];

    let j_log_t = if log_t < tbl.log_ts[1] {
        0
    } else {
        let mut found = NUM_LOG_TS - 2;
        for k in 1..(NUM_LOG_TS - 1) {
            if log_t >= tbl.log_ts[k] && log_t < tbl.log_ts[k + 1] {
                found = k;
                break;
            }
        }
        found
    };
    let log_t0 = tbl.log_ts[j_log_t];
    let log_t1 = tbl.log_ts[j_log_t + 1];

    do_kap_interpolations(
        f1,
        NUM_LOG_RHOS,
        NUM_LOG_TS,
        i_log_rho,
        j_log_t,
        log_rho0,
        log_rho,
        log_rho1,
        log_t0,
        log_t,
        log_t1,
    )
}

/// Evaluate the standard Potekhin electron conduction opacity and its logarithmic
/// derivatives (`do_electron_conduction_potekhin` in `condint.f90`).
///
/// # Errors
/// Returns [`KapError::NumericalFailure`] if `zlog` lookup fails.
pub fn do_electron_conduction_potekhin(
    zbar: f64,
    log_rho_in: f64,
    log_t_in: f64,
) -> Result<(f64, f64, f64), KapError> {
    let tbl = potekhin_table();

    let (log_rho, clipped_log_rho) = if log_rho_in < tbl.log_rhos[0] {
        (tbl.log_rhos[0], true)
    } else if log_rho_in > tbl.log_rhos[NUM_LOG_RHOS - 1] {
        (tbl.log_rhos[NUM_LOG_RHOS - 1], true)
    } else {
        (log_rho_in, false)
    };

    let (log_t, clipped_log_t) = if log_t_in < tbl.log_ts[0] {
        (tbl.log_ts[0], true)
    } else if log_t_in > tbl.log_ts[NUM_LOG_TS - 1] {
        (tbl.log_ts[NUM_LOG_TS - 1], true)
    } else {
        (log_t_in, false)
    };

    let zlog = zbar
        .max(1.0e-30)
        .log10()
        .clamp(tbl.log_zs[0], tbl.log_zs[NUM_LOG_ZS - 1]);

    let (log_k, mut dlog_k_dlog_rho, mut dlog_k_dlog_t) = if zlog
        <= tbl.log_zs[0]
    {
        eval_potekhin_z_slice(tbl, 0, log_rho, log_t)
    } else if zlog >= tbl.log_zs[NUM_LOG_ZS - 1] {
        eval_potekhin_z_slice(tbl, NUM_LOG_ZS - 1, log_rho, log_t)
    } else {
        let mut iz1 = None;
        for iz in 1..NUM_LOG_ZS {
            if zlog >= tbl.log_zs[iz - 1] && zlog <= tbl.log_zs[iz] {
                iz1 = Some(iz - 1);
                break;
            }
        }
        let i1 = iz1.ok_or(KapError::NumericalFailure("zlog lookup failed"))?;
        let i2 = i1 + 1;
        let (lk1, dlk1_dr, dlk1_dt) =
            eval_potekhin_z_slice(tbl, i1, log_rho, log_t);
        let (lk2, dlk2_dr, dlk2_dt) =
            eval_potekhin_z_slice(tbl, i2, log_rho, log_t);
        let alfa = (zlog - tbl.log_zs[i1]) / (tbl.log_zs[i2] - tbl.log_zs[i1]);
        let beta = 1.0 - alfa;
        (
            alfa * lk2 + beta * lk1,
            alfa * dlk2_dr + beta * dlk1_dr,
            alfa * dlk2_dt + beta * dlk1_dt,
        )
    };

    if clipped_log_rho {
        dlog_k_dlog_rho = 0.0;
    }
    if clipped_log_t {
        dlog_k_dlog_t = 0.0;
    }

    let logkap = 3.0 * log_t_in - log_rho_in - log_k
        + (16.0 * BOLTZ_SIGMA / 3.0).log10();
    let kap = exp10(logkap);
    let dlnkap_dln_rho = -1.0 - dlog_k_dlog_rho;
    let dlnkap_dln_t = 3.0 - dlog_k_dlog_t;
    Ok((kap, dlnkap_dln_rho, dlnkap_dln_t))
}

/// Evaluate electron conduction opacity with the Blouin et al. (2020) moderate-coupling
/// H/He correction (`do_electron_conduction_blouin` in `condint.f90`).
///
/// # Errors
/// Returns [`KapError`] if the underlying Potekhin interpolation fails.
pub fn do_electron_conduction_blouin(
    zbar: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    const ALPHA_H: f64 = -0.52;
    const ALPHA_HE: f64 = -0.46;
    const A_H: f64 = 2.0;
    const A_HE: f64 = 1.25;
    const B_H: f64 = 10.0;
    const B_HE: f64 = 2.5;
    const LOG_RHO0_H: f64 = 5.45;
    const LOG_RHO0_HE: f64 = 6.50;
    const LOG_T0_H: f64 = 8.40;
    const LOG_T0_HE: f64 = 8.57;
    const SIG_RHO_H: f64 = 5.14;
    const SIG_RHO_HE: f64 = 6.20;
    const SIG_T_H: f64 = 0.45;
    const SIG_T_HE: f64 = 0.55;

    let (mut kap, mut dlnkap_dln_rho, mut dlnkap_dln_t) =
        do_electron_conduction_potekhin(zbar, log_rho, log_t)?;

    let (frac_h, frac_he) = if zbar <= 1.0 {
        (1.0, 0.0)
    } else if zbar <= 2.0 {
        let alfa = zbar.log10() / 2.0_f64.log10();
        (1.0 - alfa, alfa)
    } else if zbar <= 3.0 {
        let log2 = 2.0_f64.log10();
        let alfa = (zbar.log10() - log2) / (3.0_f64.log10() - log2);
        (0.0, 1.0 - alfa)
    } else {
        return Ok((kap, dlnkap_dln_rho, dlnkap_dln_t));
    };

    // Independent variables: var1 = lnRho, var2 = lnT
    let log_rho_ad = AutoDiffReal2VarOrder1 {
        val: log_rho,
        d1: [ILN10, 0.0],
    };
    let log_t_ad = AutoDiffReal2VarOrder1 {
        val: log_t,
        d1: [0.0, ILN10],
    };

    let blouin_corr = |frac: f64,
                       log_rho0: f64,
                       log_t0: f64,
                       alpha: f64,
                       sig_t: f64,
                       sig_rho: f64,
                       a: f64,
                       b: f64| {
        if frac > 0.0 {
            let rhostar = log_rho_ad - log_rho0;
            let tstar = log_t_ad - log_t0;
            let (sin_a, cos_a) = alpha.sin_cos();
            let u1 = (tstar * cos_a + rhostar * sin_a) / sig_t;
            let u2 = (tstar * sin_a - rhostar * cos_a) / sig_rho;
            let g = (-(u1 * u1) - (u2 * u2)).exp() * a;
            let h = ((g - 0.5) * b).tanh() * 0.5 + 0.5;
            -((g * h + 1.0).ln())
        } else {
            AutoDiffReal2VarOrder1::from(0.0)
        }
    };

    let log_corr_h = blouin_corr(
        frac_h, LOG_RHO0_H, LOG_T0_H, ALPHA_H, SIG_T_H, SIG_RHO_H, A_H, B_H,
    );
    let log_corr_he = blouin_corr(
        frac_he,
        LOG_RHO0_HE,
        LOG_T0_HE,
        ALPHA_HE,
        SIG_T_HE,
        SIG_RHO_HE,
        A_HE,
        B_HE,
    );

    let log_correction = log_corr_h * frac_h + log_corr_he * frac_he;
    kap = (kap.ln() + log_correction.val).exp();
    dlnkap_dln_rho += log_correction.d1[0];
    dlnkap_dln_t += log_correction.d1[1];
    Ok((kap, dlnkap_dln_rho, dlnkap_dln_t))
}

/// Dispatch electron thermal conduction opacity evaluation according to [`KapConfig`]
/// (`do_electron_conduction` in `condint.f90`).
///
/// # Errors
/// Returns [`KapError`] if the underlying Potekhin interpolation fails.
pub fn do_electron_conduction(
    cfg: &KapConfig,
    zbar: f64,
    log_rho: f64,
    log_t: f64,
) -> Result<(f64, f64, f64), KapError> {
    if cfg.use_blouin_conductive_opacities {
        do_electron_conduction_blouin(zbar, log_rho, log_t)
    } else {
        do_electron_conduction_potekhin(zbar, log_rho, log_t)
    }
}
