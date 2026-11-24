// SPDX-License-Identifier: GPL-3.0-or-later
//! Dedicated `NumericsMode` verification for `flexmesa-atm` table splines
//! (`issue #210`, `issue #420`, `DECISIONS #43`).

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]

use flexmesa_atm::{atm_teff, eval_table, ATM_TABLE_PHOTOSPHERE};
use flexmesa_const::{LSUN, MSUN, RSUN, STANDARD_CGRAV};
use flexmesa_math::{log, with_numerics_mode, NumericsMode};

#[test]
fn test_eval_table_fast_mode_partials_match_centered_finite_differences() {
    let cgrav = STANDARD_CGRAV;
    let m = MSUN;
    let r = RSUN;
    let l = LSUN;
    let teff = atm_teff(l, r);
    let z_sol = flexmesa_atm::table::GN93_ZSOL;

    with_numerics_mode(NumericsMode::Fast, || {
        let out = eval_table(
            l,
            r,
            m,
            cgrav,
            ATM_TABLE_PHOTOSPHERE,
            z_sol,
            false,
            teff,
        )
        .unwrap();
        let eps = 1.0e-5;
        let r_plus = r * (1.0 + eps);
        let r_minus = r * (1.0 - eps);
        let out_plus = eval_table(
            l,
            r_plus,
            m,
            cgrav,
            ATM_TABLE_PHOTOSPHERE,
            z_sol,
            true,
            atm_teff(l, r_plus),
        )
        .unwrap();
        let out_minus = eval_table(
            l,
            r_minus,
            m,
            cgrav,
            ATM_TABLE_PHOTOSPHERE,
            z_sol,
            true,
            atm_teff(l, r_minus),
        )
        .unwrap();
        let dln_r = log(r_plus) - log(r_minus);
        let fd_pressure = (out_plus.ln_p - out_minus.ln_p) / dln_r;
        let fd_temp = (out_plus.ln_t - out_minus.ln_t) / dln_r;
        assert!(((out.dln_p_dln_r - fd_pressure) / fd_pressure).abs() < 1.0e-6);
        assert!(((out.dln_t_dln_r - fd_temp) / fd_temp).abs() < 1.0e-6);
    });
}
