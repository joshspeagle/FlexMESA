// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar wind mass-loss prescriptions (`Reimers`, `Blocker`, `Vink`,
//! `Nieuwenhuijzen`, `DeJager`, `VanLoon`, `Dutch`), 3-region pseudo-Lagrangian
//! mass adjustment (`do_adjust_mass`), conservative species overlap
//! integration, and compressional mass-change heating ($\epsilon_{\dot{M}}$)
//! corresponding to `star/private/winds.f90`, `adjust_mass.f90`,
//! `eps_mdot.f90`, `adjust_xyz.f90`, and `remove_shells.f90`.

use flexmesa_const::{CLIGHT, CRAD, L_SUN, M_SUN, PI, R_SUN, SECYER};

use crate::controls::StarError;
use crate::micro::set_basic_vars;
use crate::model::StarModel;

/// Stellar wind mass-loss prescription (`cool_wind_RGB_scheme`,
/// `cool_wind_AGB_scheme`, and `hot_wind_scheme` in `winds.f90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindScheme {
    /// Zero wind mass loss (`""`).
    None,
    /// Reimers (1975) red-giant wind (`'Reimers'`).
    Reimers,
    /// Blöcker (1995) AGB superwind (`'Blocker'`).
    Blocker,
    /// Vink, de Koter & Lamers (2001) OB-star line-driven wind (`'Vink'`).
    Vink,
    /// Nieuwenhuijzen & de Jager (1990) empirical wind (`'Nieuwenhuijzen'`).
    Nieuwenhuijzen,
    /// de Jager, Nieuwenhuijzen & van der Hucht (1988) cool wind (`'de Jager'`).
    DeJager,
    /// van Loon et al. (2005) dust-driven supergiant/AGB wind (`'van Loon'`).
    VanLoon,
    /// Composite Dutch wind (`'Dutch'`: Vink / Nugis-Lamers WR at high
    /// $T_{\mathrm{eff}}$ blended with cool wind below $10^4\,\mathrm{K}$).
    Dutch,
}

/// Low-$T_{\mathrm{eff}}$ prescription inside the composite [`WindScheme::Dutch`]
/// scheme (`Dutch_wind_lowT_scheme` in `winds.f90:415-426`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DutchLowTScheme {
    /// de Jager et al. (1988) cool wind (`'de Jager'`, default).
    #[default]
    DeJager,
    /// van Loon et al. (2005) dust-driven wind (`'van Loon'`).
    VanLoon,
    /// Nieuwenhuijzen & de Jager (1990) wind (`'Nieuwenhuijzen'`).
    Nieuwenhuijzen,
}

/// Evaluates the Reimers (1975) wind mass-loss rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`winds.f90:323, 363-364`):
/// $$w_{\mathrm{Reimers}} = \eta_{\mathrm{Reimers}}
///   \times 4\times 10^{-13}\,\frac{(L/L_\odot)(R/R_\odot)}{M/M_\odot}.$$
#[must_use]
#[inline]
pub fn eval_reimers_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || r_surf <= 0.0 {
        return 0.0;
    }
    eta * 4.0e-13 * (l_surf / L_SUN) * (r_surf / R_SUN) / (m_surf / M_SUN)
}

/// Evaluates the Blöcker (1995) wind mass-loss rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_blocker_wind` in `winds.f90:556-561`):
/// $$w_{\mathrm{Blocker}} = \eta_{\mathrm{Blocker}} \times 4.83\times 10^{-9}
///   \left(\frac{M}{M_\odot}\right)^{-2.1}
///   \left(\frac{L}{L_\odot}\right)^{2.7} w_{\mathrm{Reimers}}(\eta=1).$$
#[must_use]
#[inline]
pub fn eval_blocker_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || r_surf <= 0.0 {
        return 0.0;
    }
    let w_reimers_1 = eval_reimers_wind(l_surf, m_surf, r_surf, 1.0);
    eta * 4.83e-9
        * (m_surf / M_SUN).powf(-2.1)
        * (l_surf / L_SUN).powf(2.7)
        * w_reimers_1
}

/// Evaluates the Vink, de Koter & Lamers (2001) line-driven hot-star wind rate
/// $w$ in $M_\odot\,\mathrm{yr}^{-1}$ (`eval_Vink_wind` in `winds.f90:497-553`).
#[must_use]
pub fn eval_vink_wind(
    l_surf: f64,
    m_surf: f64,
    t_surf: f64,
    z_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0
        || l_surf <= 0.0
        || m_surf <= 0.0
        || t_surf <= 0.0
        || z_surf <= 0.0
    {
        return 0.0;
    }
    let z_solar = 0.019;
    let z_rat = (z_surf / z_solar).max(1.0e-10);
    let log_z_rat = z_rat.log10();

    // Bi-stability jump temperature (`winds.f90:509-520`).
    let d_t = 100.0;
    let t_jump = 1.0e3 * (61.2 + 2.59 * (-13.636 + 0.889 * log_z_rat));
    let alpha = if t_surf > 27_500.0 {
        1.0
    } else if t_surf < 22_500.0 {
        0.0
    } else {
        ((t_surf - (t_jump - d_t)) / (2.0 * d_t)).clamp(0.0, 1.0)
    };

    let log_l_1e5 = (l_surf / L_SUN / 1.0e5).log10();
    let log_m_30 = (m_surf / M_SUN / 30.0).log10();

    // Hot side (`w1`, Vink et al. 2001 Eq. 24).
    let vinf_hot = 2.6 * z_rat.powf(0.13);
    let log_t_4e4 = (t_surf / 4.0e4).log10();
    let log_w1 = -6.697 + 2.194 * log_l_1e5
        - 1.313 * log_m_30
        - 1.226 * (vinf_hot / 2.0).log10()
        + 0.933 * log_t_4e4
        - 10.92 * log_t_4e4 * log_t_4e4
        + 0.85 * log_z_rat;

    // Cool side (`w2`, Vink et al. 2001 Eq. 25).
    let vinf_cool = 1.3 * z_rat.powf(0.13);
    let log_t_2e4 = (t_surf / 2.0e4).log10();
    let log_w2 = -6.688 + 2.210 * log_l_1e5
        - 1.339 * log_m_30
        - 1.601 * (vinf_cool / 2.0).log10()
        + 1.07 * log_t_2e4
        + 0.85 * log_z_rat;

    eta * (alpha * 10.0_f64.powf(log_w1)
        + (1.0 - alpha) * 10.0_f64.powf(log_w2))
}

/// Evaluates the Nieuwenhuijzen & de Jager (1990) wind rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_Nieuwenhuijzen_wind` in
/// `winds.f90:619-632`).
#[must_use]
#[inline]
pub fn eval_nieuwenhuijzen_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || r_surf <= 0.0 {
        return 0.0;
    }
    let log_w = -14.02
        + 1.24 * (l_surf / L_SUN).log10()
        + 0.16 * (m_surf / M_SUN).log10()
        + 0.81 * (r_surf / R_SUN).log10();
    eta * 10.0_f64.powf(log_w)
}

/// Evaluates the de Jager et al. (1988) wind rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_de_Jager_wind` in `winds.f90:596-606`).
#[must_use]
#[inline]
pub fn eval_de_jager_wind(l_surf: f64, t_surf: f64, eta: f64) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || t_surf <= 0.0 {
        return 0.0;
    }
    let log_w =
        1.769 * (l_surf / L_SUN).log10() - 1.676 * t_surf.log10() - 8.158;
    eta * 10.0_f64.powf(log_w)
}

/// Evaluates the van Loon et al. (2005) wind rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_van_Loon_wind` in `winds.f90:609-616`).
#[must_use]
#[inline]
pub fn eval_van_loon_wind(l_surf: f64, t_surf: f64, eta: f64) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || t_surf <= 0.0 {
        return 0.0;
    }
    let log_w = -5.65 + 1.05 * (l_surf / (1.0e4 * L_SUN)).log10()
        - 6.3 * (t_surf / 3.5e3).log10();
    eta * 10.0_f64.powf(log_w)
}

/// Evaluates the Nugis & Lamers (2000) Wolf-Rayet wind mass-loss rate $w$ in
/// $M_\odot\,\mathrm{yr}^{-1}$ (`eval_highT_Dutch` in `winds.f90:459-462`):
/// $$\log_{10} w_{\mathrm{NL}} = -11.0 + 1.29 \log_{10}(L/L_\odot)
///   + 1.7 \log_{10} Y + 0.5 \log_{10} Z.$$
#[must_use]
#[inline]
pub fn eval_nugis_lamers_wind(
    l_surf: f64,
    y_surf: f64,
    z_surf: f64,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || y_surf <= 0.0 || z_surf <= 0.0 {
        return 0.0;
    }
    let log_w = -11.0
        + 1.29 * (l_surf / L_SUN).log10()
        + 1.7 * y_surf.max(1.0e-10).log10()
        + 0.5 * z_surf.max(1.0e-10).log10();
    eta * 10.0_f64.powf(log_w)
}

/// Evaluates the composite Dutch wind rate $w$ in $M_\odot\,\mathrm{yr}^{-1}$
/// (`eval_Dutch_wind` and `eval_highT_Dutch` in `winds.f90:407-481`).
#[must_use]
pub fn eval_dutch_wind(
    l_surf: f64,
    m_surf: f64,
    r_surf: f64,
    t_surf: f64,
    x_surf: f64,
    y_surf: f64,
    z_surf: f64,
    low_t_scheme: DutchLowTScheme,
    eta: f64,
) -> f64 {
    if eta <= 0.0 || l_surf <= 0.0 || m_surf <= 0.0 || t_surf <= 0.0 {
        return 0.0;
    }
    let w_high = if x_surf < 0.4 {
        let w_nl = eval_nugis_lamers_wind(l_surf, y_surf, z_surf, eta);
        if x_surf < 0.2 {
            w_nl
        } else {
            let w_vink = eval_vink_wind(l_surf, m_surf, t_surf, z_surf, eta);
            let alfa = (x_surf - 0.2) / 0.2;
            (1.0 - alfa) * w_nl + alfa * w_vink
        }
    } else {
        eval_vink_wind(l_surf, m_surf, t_surf, z_surf, eta)
    };

    let w_low = match low_t_scheme {
        DutchLowTScheme::DeJager => eval_de_jager_wind(l_surf, t_surf, eta),
        DutchLowTScheme::VanLoon => eval_van_loon_wind(l_surf, t_surf, eta),
        DutchLowTScheme::Nieuwenhuijzen => {
            eval_nieuwenhuijzen_wind(l_surf, m_surf, r_surf, eta)
        }
    };

    let t_high = 11_000.0;
    let t_low = 10_000.0;
    if t_surf >= t_high {
        w_high
    } else if t_surf <= t_low {
        w_low
    } else {
        let alfa = (t_surf - t_low) / (t_high - t_low);
        (1.0 - alfa) * w_low + alfa * w_high
    }
}

/// Evaluates the stellar wind mass-loss rate $\dot{M} \le 0$ in
/// $\mathrm{g\,s^{-1}}$ and stores it in `model.mstar_dot` (`set_mdot` in
/// `winds.f90:42-395`).
pub fn set_mdot(
    model: &mut StarModel,
    scheme: WindScheme,
    dutch_low_t: DutchLowTScheme,
    eta: f64,
) -> f64 {
    if model.nz == 0 || scheme == WindScheme::None || eta <= 0.0 {
        model.mstar_dot = 0.0;
        return 0.0;
    }
    let l_surf = model.lum[0].max(1.0);
    let m_surf = model.m_grav[0].max(1.0);
    let r_surf = model.r[0].max(1.0);
    let t_surf = if model.teff > 0.0 {
        model.teff
    } else {
        (l_surf / (PI * r_surf * r_surf * CRAD * CLIGHT)).powf(0.25)
    };
    let x_surf = model.xh_mass.first().copied().unwrap_or(0.70);
    let y_surf = model.yhe_mass.first().copied().unwrap_or(0.28);
    let z_surf = (1.0 - x_surf - y_surf).max(0.0);

    let w_msun_yr = match scheme {
        WindScheme::None => 0.0,
        WindScheme::Reimers => eval_reimers_wind(l_surf, m_surf, r_surf, eta),
        WindScheme::Blocker => eval_blocker_wind(l_surf, m_surf, r_surf, eta),
        WindScheme::Vink => eval_vink_wind(l_surf, m_surf, t_surf, z_surf, eta),
        WindScheme::Nieuwenhuijzen => {
            eval_nieuwenhuijzen_wind(l_surf, m_surf, r_surf, eta)
        }
        WindScheme::DeJager => eval_de_jager_wind(l_surf, t_surf, eta),
        WindScheme::VanLoon => eval_van_loon_wind(l_surf, t_surf, eta),
        WindScheme::Dutch => eval_dutch_wind(
            l_surf,
            m_surf,
            r_surf,
            t_surf,
            x_surf,
            y_surf,
            z_surf,
            dutch_low_t,
            eta,
        ),
    };
    let mdot_cgs = -w_msun_yr * M_SUN / SECYER;
    model.mstar_dot = mdot_cgs;
    mdot_cgs
}

/// Applies the 3-region pseudo-Lagrangian mass adjustment (`do_adjust_mass` in
/// `star/private/adjust_mass.f90:73-1131` and `eps_mdot.f90`):
/// - Region I ($0 \le k < k_A$): constant $\Delta q_{\mathrm{new}} = \Delta q_{\mathrm{old}}$.
/// - Region II ($k_A \le k < k_B$): $\Delta q_{\mathrm{new}} = q_{\mathrm{frac}}\,\Delta q_{\mathrm{old}}$.
/// - Region III ($k_B \le k < \mathrm{nz}$): constant physical mass $\Delta m_k$.
///
/// Redistributes species mass fractions `xa` via exact piecewise-constant
/// overlap integration across old mass intervals, reconstructs `ln_r` from
/// `lnd` and `dm`, and zeros `eps_mdot` (gravothermal heating is evaluated via
/// $\epsilon_{\mathrm{grav}} = -T\,ds/dt$ in `set_hydro_vars`).
///
/// # Errors
/// Returns [`StarError`] if the new envelope mass is non-positive or
/// microphysics fails.
pub fn do_adjust_mass(
    model: &mut StarModel,
    delta_m: f64,
    x_accreted: Option<&[f64]>,
    q_below_const_q: f64,
    q_const_mass: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    if let Some(x_acc) = x_accreted {
        if x_acc.len() < species {
            return Err(StarError::InvalidInput(format!(
                "x_accreted length {} < species {species}",
                x_acc.len()
            )));
        }
    }
    if delta_m == 0.0 || nz < 3 {
        model.eps_mdot.fill(0.0);
        return Ok(());
    }
    let xm_old = model.xmstar;
    let xm_new = xm_old + delta_m;
    if xm_new <= 0.0 {
        return Err(StarError::InvalidInput(format!(
            "do_adjust_mass produced non-positive envelope mass xm_new={xm_new}"
        )));
    }
    let mold_o_mnew = xm_old / xm_new;
    let mnew_o_mold = xm_new / xm_old;
    let dq_acc = delta_m / xm_new;

    // Locate 3-region indices `k_a` and `k_b` (`find_ka_and_kb`).
    let mut k_a = 1usize;
    while k_a + 2 < nz && model.q[k_a] > q_below_const_q {
        k_a += 1;
    }
    let mut k_b = (k_a + 1).min(nz - 1);
    while k_b + 1 < nz && model.q[k_b] > q_const_mass {
        k_b += 1;
    }

    let dq_old = model.dq.clone();
    let xq_old = model.xq.clone();
    let xa_old = model.xa.clone();

    // Revise `dq` across Regions I, II, and III (`revise_q_and_dq`).
    let sum_i: f64 = dq_old[..k_a].iter().sum();
    let sum_ii_old: f64 = dq_old[k_a..k_b].iter().sum();
    let sum_iii_old: f64 = dq_old[k_b..].iter().sum();
    let sum_iii_new = sum_iii_old * mold_o_mnew;
    let sum_ii_new = (1.0 - sum_i - sum_iii_new).max(0.0);
    let q_frac = (sum_ii_new / sum_ii_old.max(1.0e-99)).max(1.0e-12);

    model.dq[..k_a].copy_from_slice(&dq_old[..k_a]);
    for k in k_a..k_b {
        model.dq[k] = q_frac * dq_old[k];
    }
    for k in k_b..nz {
        model.dq[k] = dq_old[k] * mold_o_mnew;
    }
    if k_b > k_a {
        let sum_before_last: f64 = model.dq[..(k_b - 1)].iter().sum();
        let target_kb = 1.0 - sum_iii_new;
        if target_kb > sum_before_last {
            model.dq[k_b - 1] = target_kb - sum_before_last;
        }
    }

    model.mstar += delta_m;
    model.xmstar = xm_new;
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();

    // Conservative piecewise-constant overlap projection (`set_xa`,
    // `adjust_mass.f90:878-1131`).
    let default_acc: Vec<f64> = model.cell_xa(0).to_vec();
    let x_acc = x_accreted.unwrap_or(&default_acc);
    let mut accum = vec![0.0; species];
    let mut kk_start = 0_usize;
    for k in 0..k_b {
        let xq_00 = model.xq[k];
        let xq_p1 = xq_00 + model.dq[k];
        let x_00 = (xq_00 - dq_acc) * mnew_o_mold;
        let x_p1 = (xq_p1 - dq_acc) * mnew_o_mold;
        let dx = (x_p1 - x_00).max(1.0e-99);

        if x_p1 <= 0.0 {
            model.cell_xa_mut(k).copy_from_slice(&x_acc[..species]);
            model.eps_mdot[k] = 0.0;
        } else {
            accum.fill(0.0);
            if x_00 < 0.0 {
                let w_acc = -x_00;
                for j in 0..species {
                    accum[j] += w_acc * x_acc[j];
                }
            }
            let a_start = x_00.max(0.0);
            while kk_start + 1 < nz
                && xq_old[kk_start] + dq_old[kk_start] <= a_start
            {
                kk_start += 1;
            }
            for kk in kk_start..nz {
                let b0 = xq_old[kk];
                if b0 >= x_p1 {
                    break;
                }
                let b1 = b0 + dq_old[kk];
                let overlap = (x_p1.min(b1) - a_start.max(b0)).max(0.0);
                if overlap > 0.0 {
                    for j in 0..species {
                        accum[j] += overlap * xa_old[kk * species + j];
                    }
                }
            }
            let mut sum_x = 0.0;
            for j in 0..species {
                let xj = (accum[j] / dx).clamp(0.0, 1.0);
                model.xa[k * species + j] = xj;
                sum_x += xj;
            }
            if sum_x > 0.0 {
                for j in 0..species {
                    model.xa[k * species + j] /= sum_x;
                }
            }
            model.eps_mdot[k] = 0.0;
        }
    }
    for k in k_b..nz {
        model.eps_mdot[k] = 0.0;
    }

    model.reconstruct_ln_r_from_lnd_and_dm();

    set_basic_vars(model)
}
