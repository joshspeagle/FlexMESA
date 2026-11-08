// SPDX-License-Identifier: LGPL-3.0-or-later
//! Thin accretion disk outer $L_2$ mass-loss fraction following Lu et al.
//! (2023, MNRAS 519, 1409).
//!
//! Corresponds to `binary/private/binary_disk.f90:43-316`.

use flexmesa_const::{
    BOLTZM, CLIGHT, CRAD, MP, MSUN, ONE_THIRD, PI, PI2, RSUN, SECYER,
    STANDARD_CGRAV, TWO_THIRDS,
};
use flexmesa_math::{exp10, log10, pow};

const ONE_FOURTH: f64 = 1.0 / 4.0;
const SEVEN_FOURTHS: f64 = 7.0 / 4.0;
const ONE_EIGHTH: f64 = 1.0 / 8.0;
const THREE_EIGHTHS: f64 = 3.0 / 8.0;
const ONE_NINTH: f64 = 1.0 / 9.0;
const FOUR_TWENTYSEVENTHS: f64 = 4.0 / 27.0;

#[inline]
fn bisect_rel<F: Fn(f64) -> f64>(
    mut left: f64,
    mut right: f64,
    mut f_left: f64,
    tol: f64,
    f: F,
) -> f64 {
    while ((left - right) / right).abs() > tol {
        let mid = 0.5 * (left + right);
        let f_mid = f(mid);
        if f_mid * f_left > 0.0 {
            left = mid;
            f_left = f_mid;
        } else {
            right = mid;
        }
    }
    0.5 * (left + right)
}

#[inline]
fn f1_the_t_fl2(the: f64, t: f64, fl2: f64, c1: f64, c2: f64) -> f64 {
    let t2 = t * t;
    let t4 = t2 * t2;
    let the2 = the * the;
    let the3 = the2 * the;
    c1 * t4 * the3 / (1.0 - fl2) - the2 + c2 * t
}

#[inline]
fn kap_kramers(rho: f64, t: f64) -> f64 {
    0.34 + 3.0e24 * rho * pow(t, -3.5)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn f2_the_t_fl2(
    the: f64,
    t: f64,
    fl2: f64,
    c3: f64,
    c4: f64,
    m1dot: f64,
    disk_alpha: f64,
    omega_k: f64,
    rd: f64,
    phi_rd: f64,
    gm2: f64,
    phi_l1: f64,
    phi_l2: f64,
) -> f64 {
    let t_the = t * the;
    let x = c4 * (t_the * t_the * t_the) / (1.0 - fl2);
    let u_over_p = (1.5 + x) / (1.0 + ONE_THIRD * x);
    let rd3 = rd * rd * rd;
    let the2 = the * the;
    let rho =
        (1.0 - fl2) * m1dot / (2.0 * PI * disk_alpha * omega_k * rd3) / the2;
    let t2 = t * t;
    let t4 = t2 * t2;
    let one_m_fl2 = 1.0 - fl2;
    let gm2_over_rd = gm2 / rd;
    SEVEN_FOURTHS
        - (1.5 * u_over_p
            + c3 * t4 / kap_kramers(rho, t) / (one_m_fl2 * one_m_fl2))
            * the2
        - phi_rd / gm2_over_rd
        + (phi_l1 - fl2 * phi_l2) / gm2_over_rd / one_m_fl2
}

#[inline]
fn t_the_nofl2(
    the: f64,
    logthe_grid: &[f64; 50],
    logt_arr: &[f64; 50],
    dlogthe: f64,
) -> f64 {
    let logthe = log10(the);
    if logthe > logthe_grid[48] {
        exp10(logt_arr[48] - 0.25 * (logthe - logthe_grid[48]))
    } else if logthe < logthe_grid[0] {
        exp10(logt_arr[0] + 2.0 * (logthe - logthe_grid[0]))
    } else {
        let idx =
            (((logthe - logthe_grid[0]) / dlogthe).floor() as usize).min(48);
        let slope = (logt_arr[idx + 1] - logt_arr[idx]) / dlogthe;
        let logt = logt_arr[idx] + (logthe - logthe_grid[idx]) * slope;
        exp10(logt)
    }
}

#[inline]
fn t_the_nonzero(
    the: f64,
    c2: f64,
    phi_l2: f64,
    phi_rd: f64,
    gm2: f64,
    rd: f64,
) -> f64 {
    (8.0 * the * the + 1.0 - 2.0 * (phi_l2 - phi_rd) / (gm2 / rd)) / (3.0 * c2)
}

#[inline]
fn fl2_from_the_and_t(the: f64, t: f64, c1: f64, c2: f64) -> f64 {
    let t2 = t * t;
    let t4 = t2 * t2;
    let the2 = the * the;
    let the3 = the2 * the;
    1.0 - c1 * t4 * the3 / (the2 - c2 * t)
}

/// Evaluates the outer $L_2$ mass-loss fraction $f_{L2} \in [0, 1]$ for a thin
/// accretion disk around the accretor following Lu et al. (2023, MNRAS 519,
/// 1409).
///
/// Corresponds to `eval_L2_mass_loss_fraction` in
/// `binary/private/binary_disk.f90:43-316`.
#[must_use]
pub fn eval_l2_mass_loss_fraction(
    donor_mass: f64,
    accretor_mass: f64,
    mass_transfer_rate: f64,
    orbital_separation: f64,
    disk_alpha: f64,
    disk_mu: f64,
) -> f64 {
    const EPS_SMALL: f64 = 1.0e-12;
    const TOL: f64 = 1.0e-8;
    const N_THE: usize = 50;
    const THE_GRID_MIN: f64 = 0.1;
    const THE_GRID_MAX: f64 = 1.0;
    const T_FLOOR: f64 = 3.0e3;

    if mass_transfer_rate < EPS_SMALL {
        return 0.0;
    }

    let m2 = accretor_mass * MSUN;
    let m1dot = mass_transfer_rate * MSUN / SECYER;
    let a = orbital_separation * RSUN;
    let q = accretor_mass / donor_mass;
    let log_q = log10(q);

    let mut xl1 = -0.0355 * log_q * log_q + 0.251 * log_q.abs() + 0.500;
    let mut xl2 = 0.0756 * log_q * log_q - 0.424 * log_q.abs() + 1.699;
    if log_q > 0.0 {
        xl1 = 1.0 - xl1;
        xl2 = 1.0 - xl2;
    }
    let mu = q / (1.0 + q);
    let one_m_xl1 = 1.0 - xl1;
    let rd_over_a = (one_m_xl1 * one_m_xl1 * one_m_xl1 * one_m_xl1) / mu;

    let phi_l1_dimless = -((1.0 - mu) / xl1.abs()
        + mu / (1.0 - xl1).abs()
        + 0.5 * (xl1 - mu) * (xl1 - mu));
    let phi_l2_dimless = -((1.0 - mu) / xl2.abs()
        + mu / (1.0 - xl2).abs()
        + 0.5 * (xl2 - mu) * (xl2 - mu));
    let phi_rd_dimless =
        -(1.0 - mu + mu / rd_over_a + 0.5 * (1.0 - mu) * (1.0 - mu));

    let gm2 = STANDARD_CGRAV * m2;
    let rd = rd_over_a * a;
    let phi_units = STANDARD_CGRAV * (m2 / mu) / a;
    let phi_l1 = phi_l1_dimless * phi_units;
    let phi_l2 = phi_l2_dimless * phi_units;
    let phi_rd = phi_rd_dimless * phi_units;
    let rd3 = rd * rd * rd;
    let omega_k = (gm2 / rd3).sqrt();

    let c1 = TWO_THIRDS * PI * CRAD * disk_alpha * rd / (omega_k * m1dot);
    let c2 = BOLTZM * rd / (gm2 * disk_mu * MP);
    let c3 = 8.0 * PI2 * CRAD * disk_alpha * CLIGHT * (rd * rd)
        / (m1dot * m1dot * omega_k);
    let c4 = 2.0 * PI * disk_mu * CRAD * disk_alpha * omega_k * MP * rd3
        / (BOLTZM * m1dot);

    let mut the_grid = [0.0_f64; N_THE];
    let mut logt_arr = [0.0_f64; N_THE];
    let mut logthe_grid = [0.0_f64; N_THE];

    let log_min = log10(THE_GRID_MIN);
    let log_max = log10(THE_GRID_MAX);
    for (i, slot) in the_grid.iter_mut().enumerate() {
        *slot = exp10(
            log_min + (i as f64) * (log_max - log_min) / ((N_THE - 1) as f64),
        );
    }

    let t_max = pow(FOUR_TWENTYSEVENTHS / (c1 * c1 * c2), ONE_NINTH);
    for i in 0..N_THE {
        let the = the_grid[i];
        let t_left = 0.1 * ((the * the) / c2).min(t_max);
        let f1_left = f1_the_t_fl2(the, t_left, 0.0, c1, c2);
        let t_mid = bisect_rel(t_left, t_max, f1_left, TOL, |t| {
            f1_the_t_fl2(the, t, 0.0, c1, c2)
        });
        logt_arr[i] = log10(t_mid);
        logthe_grid[i] = log10(the);
    }
    let dlogthe = logthe_grid[1] - logthe_grid[0];

    let eval_f2_zero = |the: f64| {
        let t = t_the_nofl2(the, &logthe_grid, &logt_arr, dlogthe);
        f2_the_t_fl2(
            the, t, 0.0, c3, c4, m1dot, disk_alpha, omega_k, rd, phi_rd, gm2,
            phi_l1, phi_l2,
        )
    };

    let mut the_right = 1.0_f64;
    let mut f2_right = eval_f2_zero(the_right);
    let separation_factor = 0.95_f64;
    let mut the_left = separation_factor * the_right;
    let mut f2_left = eval_f2_zero(the_left);
    while f2_left * f2_right > 0.0 {
        the_right = the_left;
        f2_right = f2_left;
        the_left *= separation_factor;
        f2_left = eval_f2_zero(the_left);
    }
    let the_sol = bisect_rel(the_left, the_right, f2_left, TOL, eval_f2_zero);
    let t_sol = t_the_nofl2(the_sol, &logthe_grid, &logt_arr, dlogthe);
    let the_max_init = (THREE_EIGHTHS * c2 * t_sol
        + ONE_FOURTH * (phi_l2 - phi_rd) / (gm2 / rd)
        - ONE_EIGHTH)
        .sqrt();

    if the_sol < the_max_init {
        EPS_SMALL
    } else {
        let the_min_zero = 0.5 * ((phi_l2 - phi_rd) / (gm2 / rd) - 0.5).sqrt();
        let f_l = t_the_nonzero(the_min_zero, c2, phi_l2, phi_rd, gm2, rd)
            - t_the_nofl2(the_min_zero, &logthe_grid, &logt_arr, dlogthe);
        let the_max = bisect_rel(the_min_zero, 1.0, f_l, TOL, |the| {
            t_the_nonzero(the, c2, phi_l2, phi_rd, gm2, rd)
                - t_the_nofl2(the, &logthe_grid, &logt_arr, dlogthe)
        });
        let the_min = (THREE_EIGHTHS * c2 * T_FLOOR
            + ONE_FOURTH * (phi_l2 - phi_rd) / (gm2 / rd)
            - ONE_EIGHTH)
            .sqrt();
        let eval_f2_nz = |the: f64| {
            let t = t_the_nonzero(the, c2, phi_l2, phi_rd, gm2, rd);
            let fl2 = fl2_from_the_and_t(the, t, c1, c2);
            f2_the_t_fl2(
                the, t, fl2, c3, c4, m1dot, disk_alpha, omega_k, rd, phi_rd,
                gm2, phi_l1, phi_l2,
            )
        };
        let f2_left2 = eval_f2_nz(the_min);
        let the_final = bisect_rel(
            the_min,
            the_max / (1.0 + EPS_SMALL),
            f2_left2,
            TOL,
            eval_f2_nz,
        );
        let t_final = t_the_nonzero(the_final, c2, phi_l2, phi_rd, gm2, rd);
        fl2_from_the_and_t(the_final, t_final, c1, c2)
    }
}
