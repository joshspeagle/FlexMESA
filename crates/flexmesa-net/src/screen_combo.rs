// SPDX-License-Identifier: LGPL-3.0-or-later
//! Compound reaction branch weighting (`set_combo_screen_rates` and
//! `rate_for_pg_pa_branches` in `net/private/net_screen.f90` lines 530–784).

use crate::types::{NetError, NetGeneralInfo, IH1};

const ALPHA_AP_BRANCHES: [(&str, &str, &str, &str); 10] = [
    ("rc12ap_aux", "rn15pg_aux", "rn15pa_aux", "rc12ap_to_o16"),
    ("ro16ap_aux", "rf19pg_aux", "rf19pa_aux", "ro16ap_to_ne20"),
    (
        "rne20ap_aux",
        "rna23pg_aux",
        "rna23pa_aux",
        "rne20ap_to_mg24",
    ),
    (
        "rmg24ap_aux",
        "ral27pg_aux",
        "ral27pa_aux",
        "rmg24ap_to_si28",
    ),
    ("rsi28ap_aux", "rp31pg_aux", "rp31pa_aux", "rsi28ap_to_s32"),
    ("rs32ap_aux", "rcl35pg_aux", "rcl35pa_aux", "rs32ap_to_ar36"),
    ("rar36ap_aux", "rk39pg_aux", "rk39pa_aux", "rar36ap_to_ca40"),
    (
        "rca40ap_aux",
        "rsc43pg_aux",
        "rsc43pa_aux",
        "rca40ap_to_ti44",
    ),
    ("rti44ap_aux", "rv47pg_aux", "rv47pa_aux", "rti44ap_to_cr48"),
    (
        "rcr48ap_aux",
        "rmn51pg_aux",
        "rmn51pa_aux",
        "rcr48ap_to_fe52",
    ),
];

const ALPHA_GP_BRANCHES: [(&str, &str, &str, &str); 10] = [
    ("ro16gp_aux", "rn15pg_aux", "rn15pa_aux", "ro16gp_to_c12"),
    ("rne20gp_aux", "rf19pg_aux", "rf19pa_aux", "rne20gp_to_o16"),
    (
        "rmg24gp_aux",
        "rna23pg_aux",
        "rna23pa_aux",
        "rmg24gp_to_ne20",
    ),
    (
        "rsi28gp_aux",
        "ral27pg_aux",
        "ral27pa_aux",
        "rsi28gp_to_mg24",
    ),
    ("rs32gp_aux", "rp31pg_aux", "rp31pa_aux", "rs32gp_to_si28"),
    (
        "rar36gp_aux",
        "rcl35pg_aux",
        "rcl35pa_aux",
        "rar36gp_to_s32",
    ),
    ("rca40gp_aux", "rk39pg_aux", "rk39pa_aux", "rca40gp_to_ar36"),
    (
        "rti44gp_aux",
        "rsc43pg_aux",
        "rsc43pa_aux",
        "rti44gp_to_ca40",
    ),
    ("rcr48gp_aux", "rv47pg_aux", "rv47pa_aux", "rcr48gp_to_ti44"),
    (
        "rfe52gp_aux",
        "rmn51pg_aux",
        "rmn51pa_aux",
        "rfe52gp_to_cr48",
    ),
];

/// Weights compound pp2/pp3 and $(\alpha, p)/(p, \gamma)$ branch rates after
/// pairwise screening (`set_combo_screen_rates` in `net/private/net_screen.f90`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn set_combo_screen_rates(
    g: &NetGeneralInfo,
    y: &[f64],
    rate_raw: &[f64],
    rate_raw_dt: &[f64],
    rate_raw_drho: &[f64],
    rate_screened: &mut [f64],
    rate_screened_dt: &mut [f64],
    rate_screened_drho: &mut [f64],
) -> Result<(), NetError> {
    let rtab = |name: &str| g.reaction_index_by_name(name).unwrap_or(0);
    let i_pp2 = rtab("r34_pp2");
    let i_pp3 = rtab("r34_pp3");
    if i_pp2 != 0 && i_pp3 != 0 {
        if (rate_screened[i_pp2 - 1] - rate_screened[i_pp3 - 1]).abs() > 0.0 {
            return Err(NetError::IntegrationError(
                "r34_pp2 and r34_pp3 screened rates differ before branching"
                    .to_string(),
            ));
        }
        let rate_ii = if rtab("r_be7_wk_li7") != 0 {
            rate_screened[rtab("r_be7_wk_li7") - 1]
        } else if rtab("rbe7ec_li7_aux") != 0 {
            rate_screened[rtab("rbe7ec_li7_aux") - 1]
        } else {
            return Err(NetError::MissingReaction(
                "rbe7ec_li7_aux".to_string(),
            ));
        };
        let yh1 = y[g.net_iso[IH1] - 1];
        let rate_iii = if rtab("r_be7_pg_b8") != 0 {
            yh1 * rate_screened[rtab("r_be7_pg_b8") - 1]
        } else if rtab("rbe7pg_b8_aux") != 0 {
            yh1 * rate_screened[rtab("rbe7pg_b8_aux") - 1]
        } else {
            return Err(NetError::MissingReaction("rbe7pg_b8_aux".to_string()));
        };
        let rsum = rate_ii + rate_iii;
        let f_ii = if rsum < 1.0e-50 { 0.5 } else { rate_ii / rsum };
        let f_iii = 1.0 - f_ii;

        rate_screened[i_pp2 - 1] *= f_ii;
        rate_screened_dt[i_pp2 - 1] *= f_ii;
        rate_screened_drho[i_pp2 - 1] *= f_ii;

        rate_screened[i_pp3 - 1] *= f_iii;
        rate_screened_dt[i_pp3 - 1] *= f_iii;
        rate_screened_drho[i_pp3 - 1] *= f_iii;
    }

    if rtab("rn14_to_c12") != 0 {
        rate_for_pg_pa_branches(
            rtab("rn14pg_aux"),
            rtab("rn15pg_aux"),
            rtab("rn15pa_aux"),
            0,
            rtab("rn14_to_c12"),
            rate_raw,
            rate_raw_dt,
            rate_raw_drho,
            rate_screened,
            rate_screened_dt,
            rate_screened_drho,
        );
    }
    if rtab("rn14_to_o16") != 0 {
        rate_for_pg_pa_branches(
            rtab("rn14pg_aux"),
            rtab("rn15pg_aux"),
            rtab("rn15pa_aux"),
            rtab("rn14_to_o16"),
            0,
            rate_raw,
            rate_raw_dt,
            rate_raw_drho,
            rate_screened,
            rate_screened_dt,
            rate_screened_drho,
        );
    }
    if rtab("r1616ppa") != 0 {
        rate_for_pg_pa_branches(
            rtab("r1616p_aux"),
            rtab("rp31pg_aux"),
            rtab("rp31pa_aux"),
            0,
            rtab("r1616ppa"),
            rate_raw,
            rate_raw_dt,
            rate_raw_drho,
            rate_screened,
            rate_screened_dt,
            rate_screened_drho,
        );
    }
    if rtab("r1616ppg") != 0 {
        rate_for_pg_pa_branches(
            rtab("r1616p_aux"),
            rtab("rp31pg_aux"),
            rtab("rp31pa_aux"),
            rtab("r1616ppg"),
            0,
            rate_raw,
            rate_raw_dt,
            rate_raw_drho,
            rate_screened,
            rate_screened_dt,
            rate_screened_drho,
        );
    }

    for (start, pg, pa, with_pg) in ALPHA_AP_BRANCHES {
        let ir_with_pg = rtab(with_pg);
        if ir_with_pg != 0 {
            rate_for_pg_pa_branches(
                rtab(start),
                rtab(pg),
                rtab(pa),
                ir_with_pg,
                0,
                rate_raw,
                rate_raw_dt,
                rate_raw_drho,
                rate_screened,
                rate_screened_dt,
                rate_screened_drho,
            );
        }
    }

    for (start, pg, pa, with_pa) in ALPHA_GP_BRANCHES {
        let ir_with_pa = rtab(with_pa);
        if ir_with_pa != 0 {
            rate_for_pg_pa_branches(
                rtab(start),
                rtab(pg),
                rtab(pa),
                0,
                ir_with_pa,
                rate_raw,
                rate_raw_dt,
                rate_raw_drho,
                rate_screened,
                rate_screened_dt,
                rate_screened_drho,
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rate_for_pg_pa_branches(
    ir_start: usize,
    irpg: usize,
    irpa: usize,
    ir_with_pg: usize,
    ir_with_pa: usize,
    rate_raw: &[f64],
    rate_raw_dt: &[f64],
    rate_raw_drho: &[f64],
    rate_screened: &mut [f64],
    rate_screened_dt: &mut [f64],
    rate_screened_drho: &mut [f64],
) {
    let mut pg_raw_rate = rate_raw[irpg - 1];
    let mut pa_raw_rate = rate_raw[irpa - 1];
    if pg_raw_rate + pa_raw_rate < 1.0e-99 {
        pg_raw_rate = 1.0;
        pa_raw_rate = 1.0;
    }
    let pg_frac = pg_raw_rate / (pg_raw_rate + pa_raw_rate);
    let pa_frac = 1.0 - pg_frac;
    let x = pg_raw_rate + pa_raw_rate;
    let d_pg_frac_dt = (pa_raw_rate * rate_raw_dt[irpg - 1]
        - pg_raw_rate * rate_raw_dt[irpa - 1])
        / (x * x);
    let d_pa_frac_dt = -d_pg_frac_dt;
    let d_pg_frac_drho = (pa_raw_rate * rate_raw_drho[irpg - 1]
        - pg_raw_rate * rate_raw_drho[irpa - 1])
        / (x * x);
    let d_pa_frac_drho = -d_pg_frac_drho;

    let r = rate_screened[ir_start - 1];
    let drdt = rate_screened_dt[ir_start - 1];
    let drdd = rate_screened_drho[ir_start - 1];

    if ir_with_pg != 0 {
        rate_screened[ir_with_pg - 1] = r * pg_frac;
        rate_screened_dt[ir_with_pg - 1] = r * d_pg_frac_dt + drdt * pg_frac;
        rate_screened_drho[ir_with_pg - 1] =
            r * d_pg_frac_drho + drdd * pg_frac;
    }
    if ir_with_pa != 0 {
        rate_screened[ir_with_pa - 1] = r * pa_frac;
        rate_screened_dt[ir_with_pa - 1] = r * d_pa_frac_dt + drdt * pa_frac;
        rate_screened_drho[ir_with_pa - 1] =
            r * d_pa_frac_drho + drdd * pa_frac;
    }
}
