// SPDX-License-Identifier: LGPL-3.0-or-later
//! Control-driven stellar wind and mass-change driver (`set_mdot`,
//! `do_set_mdot`) matching `star/private/winds.f90:54-476`.

use flexmesa_const::{
    CLIGHT, CRAD, FOUR_PI, M_SUN, PI, SECYER, STANDARD_CGRAV,
};

use crate::controls::StarError;
use crate::model::StarModel;
use crate::winds::{eval_scheme_rate, DutchLowTScheme, WindScheme};

#[inline]
fn surface_teff(model: &StarModel, l_surf: f64, r_surf: f64) -> f64 {
    if model.teff > 0.0 {
        model.teff
    } else {
        (l_surf / (PI * r_surf * r_surf * CRAD * CLIGHT)).powf(0.25)
    }
}

/// Evaluates a single wind scheme and writes `model.mstar_dot` (`g s^-1`).
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
    let t_surf = surface_teff(model, l_surf, r_surf);
    let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
    let x_surf = model.xh_mass.first().copied().unwrap_or(0.70);
    let y_surf = model.yhe_mass.first().copied().unwrap_or(0.28);
    let z_surf = (1.0 - x_surf - y_surf).max(0.0);

    let w_msun_yr = eval_scheme_rate(
        scheme,
        dutch_low_t,
        l_surf,
        m_surf,
        r_surf,
        t_surf,
        x_surf,
        y_surf,
        z_surf,
        cgrav,
        eta,
    );
    let mdot_cgs = -w_msun_yr * M_SUN / SECYER;
    model.mstar_dot = mdot_cgs;
    mdot_cgs
}

fn scheme_eta(model: &StarModel, scheme: WindScheme) -> f64 {
    let c = &model.controls;
    let mw = &c.mix_wind;
    match scheme {
        WindScheme::None => 0.0,
        WindScheme::Reimers => c.reimers_scaling_factor,
        WindScheme::Blocker => c.blocker_scaling_factor,
        WindScheme::Vink => c.vink_scaling_factor,
        WindScheme::Nieuwenhuijzen => mw.nieuwenhuijzen_scaling_factor,
        WindScheme::DeJager => mw.de_jager_scaling_factor,
        WindScheme::VanLoon => mw.van_loon_scaling_factor,
        WindScheme::Bjorklund => mw.bjorklund_scaling_factor,
        WindScheme::Dutch => c.dutch_scaling_factor,
    }
}

/// Evaluates the full control-driven mass-change rate `model.mstar_dot`
/// (`g s^-1`, negative for mass loss, positive for accretion) matching
/// `do_set_mdot` in `star/private/winds.f90:54-476`.
///
/// # Errors
/// Returns [`StarError`] if `hot_wind_full_on_t < cool_wind_full_on_t` or an
/// unknown wind scheme name is configured.
pub fn do_set_mdot(model: &mut StarModel) -> Result<f64, StarError> {
    let nz = model.nz;
    if nz == 0 {
        model.mstar_dot = 0.0;
        return Ok(0.0);
    }
    let c = &model.controls;
    if c.hot_wind_full_on_t < c.cool_wind_full_on_t {
        return Err(StarError::InvalidInput(String::from(
            "hot_wind_full_on_T < cool_wind_full_on_T in do_set_mdot",
        )));
    }

    let l1 = model.lum[0].max(1.0);
    let m1 = model.m_grav[0].max(1.0);
    let t1 = surface_teff(model, l1, model.r[0].max(1.0));
    let r1 = (l1 / (PI * CRAD * CLIGHT * t1.powi(4))).sqrt().max(1.0);
    let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
    let kap_surf = model.opacity.first().copied().unwrap_or(0.34).max(1.0e-4);
    let l_edd = FOUR_PI * CLIGHT * cgrav * m1 / kap_surf;

    let surf_h1 = model.xh_mass.first().copied().unwrap_or(0.0);
    let surf_he4 = model.yhe_mass.first().copied().unwrap_or(0.0);
    let cen_h1 = model.xh_mass.last().copied().unwrap_or(0.0);
    let cen_he4 = model.yhe_mass.last().copied().unwrap_or(0.0);
    let z_surf = if c.zbase >= 0.0 {
        c.zbase
    } else {
        (1.0 - surf_h1 - surf_he4).max(0.0)
    };

    let mw = &c.mix_wind;
    let mut wind_mdot = 0.0_f64;
    let mut using_wind_scheme = false;

    // 1. Super-Eddington wind (`eval_super_eddington_wind`, `winds.f90:637-704`).
    if c.super_eddington_scaling_factor > 0.0 {
        let ledd_eff = mw.super_eddington_wind_ledd_factor.max(0.0) * l_edd;
        if l1 > ledd_eff {
            let v_esc2 = (2.0 * cgrav * m1 / r1).max(1.0);
            wind_mdot =
                c.super_eddington_scaling_factor * (l1 - ledd_eff) / v_esc2;
        }
    }

    // 2. Explicit `mass_change` (`M_sun yr^-1`, positive = accretion -> negative `wind_mdot`).
    if (c.mass_change > 0.0 && wind_mdot == 0.0)
        || (c.mass_change < 0.0 && -c.mass_change * M_SUN / SECYER > wind_mdot)
    {
        wind_mdot = -c.mass_change * M_SUN / SECYER;
    }

    // 3. Hot + cool RGB/AGB wind schemes (with legacy fallback when scheme names
    // are blank but a positive scaling factor was set directly on `controls`).
    let dutch_low = DutchLowTScheme::parse_name(&mw.dutch_wind_lowt_scheme);
    let mut hot_s =
        WindScheme::parse_name(&mw.hot_wind_scheme).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "unknown hot_wind_scheme '{}'",
                mw.hot_wind_scheme
            ))
        })?;
    let cool_name = if cen_h1 < 0.01 && cen_he4 < mw.rgb_to_agb_wind_switch {
        &c.cool_wind_agb_scheme
    } else {
        &c.cool_wind_rgb_scheme
    };
    let mut cool_s = WindScheme::parse_name(cool_name).ok_or_else(|| {
        StarError::InvalidInput(format!(
            "unknown cool_wind_scheme '{cool_name}'"
        ))
    })?;

    if hot_s == WindScheme::None && cool_s == WindScheme::None {
        let fallback = if c.dutch_scaling_factor > 0.0 {
            WindScheme::Dutch
        } else if c.vink_scaling_factor > 0.0 {
            WindScheme::Vink
        } else if mw.bjorklund_scaling_factor > 0.0 {
            WindScheme::Bjorklund
        } else if c.reimers_scaling_factor > 0.0 {
            WindScheme::Reimers
        } else if c.blocker_scaling_factor > 0.0 {
            WindScheme::Blocker
        } else {
            WindScheme::None
        };
        hot_s = fallback;
        cool_s = fallback;
    }

    let hot_w = if t1 >= c.cool_wind_full_on_t {
        eval_scheme_rate(
            hot_s,
            dutch_low,
            l1,
            m1,
            r1,
            t1,
            surf_h1,
            surf_he4,
            z_surf,
            cgrav,
            scheme_eta(model, hot_s),
        )
    } else {
        0.0
    };
    let cool_w = if t1 <= c.hot_wind_full_on_t {
        eval_scheme_rate(
            cool_s,
            dutch_low,
            l1,
            m1,
            r1,
            t1,
            surf_h1,
            surf_he4,
            z_surf,
            cgrav,
            scheme_eta(model, cool_s),
        )
    } else {
        0.0
    };

    let wind_msun_yr = if t1 >= c.hot_wind_full_on_t {
        hot_w
    } else if t1 <= c.cool_wind_full_on_t {
        cool_w
    } else if (c.hot_wind_full_on_t - c.cool_wind_full_on_t).abs() <= 1.0e-12 {
        0.5 * (hot_w + cool_w)
    } else {
        let beta = ((c.hot_wind_full_on_t - t1)
            / (c.hot_wind_full_on_t - c.cool_wind_full_on_t))
            .clamp(0.0, 1.0);
        (1.0 - beta) * hot_w + beta * cool_w
    };

    let wind_cgs = wind_msun_yr * M_SUN / SECYER;
    if wind_cgs > wind_mdot.abs() {
        using_wind_scheme = true;
        wind_mdot = wind_cgs;
    }

    if using_wind_scheme && mw.no_wind_if_no_rotation && !c.rotation_flag {
        model.mstar_dot = 0.0;
        model.aux.step_hist.rotational_mdot_boost_old = 1.0;
        return Ok(0.0);
    }

    // 4. Super-Eddington boost (`winds.f90:216-236`).
    if wind_mdot >= 0.0
        && c.super_eddington_scaling_factor <= 0.0
        && l_edd > 0.0
    {
        let l_div_ledd = l1 / l_edd;
        if l_div_ledd > mw.wind_boost_full_off_l_div_ledd {
            let boost = if l_div_ledd >= mw.wind_boost_full_on_l_div_ledd {
                mw.super_eddington_wind_max_boost
            } else {
                1.0 + (mw.super_eddington_wind_max_boost - 1.0)
                    * (l_div_ledd - mw.wind_boost_full_off_l_div_ledd)
                    / (mw.wind_boost_full_on_l_div_ledd
                        - mw.wind_boost_full_off_l_div_ledd)
                        .max(1.0e-30)
            };
            wind_mdot *= boost.max(1.0);
        }
    }

    if wind_mdot >= 0.0 && mw.min_wind > 0.0 {
        wind_mdot = wind_mdot.max(mw.min_wind * M_SUN / SECYER);
    }
    if wind_mdot >= 0.0 && mw.max_wind > 0.0 {
        wind_mdot = wind_mdot.min(mw.max_wind * M_SUN / SECYER);
    }

    let mut mdot_out = -wind_mdot;
    if mdot_out > 0.0
        && mw.max_star_mass_for_gain > 0.0
        && model.mstar >= mw.max_star_mass_for_gain * M_SUN
    {
        mdot_out = 0.0;
    }
    if mdot_out < 0.0
        && mw.min_star_mass_for_loss > 0.0
        && model.mstar <= mw.min_star_mass_for_loss * M_SUN
    {
        mdot_out = 0.0;
    }

    // 5. Heger, Langer & Woosley (2000) rotational enhancement (`winds.f90:412-476`).
    let mut rot_boost = 1.0_f64;
    if mdot_out < 0.0 && c.rotation_flag && mw.mdot_omega_power > 0.0 {
        let base_w = -mdot_out;
        let omega_crit = crate::rotation::eval_omega_crit(
            cgrav, m1, r1, l1, kap_surf, 0.999_999,
        );
        let w_div_wc = if omega_crit > 0.0 {
            (model.omega[0].abs() / omega_crit).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if mw.min_wind > 0.0 || using_wind_scheme || w_div_wc > 0.8 {
            let mut enh =
                (1.0 - w_div_wc).max(1.0e-22).powf(-mw.mdot_omega_power);
            if mw.max_rotational_mdot_boost > 0.0 {
                enh = enh.min(mw.max_rotational_mdot_boost);
            }
            let t_kh = (0.75 * cgrav * m1 * m1 / (r1 * l1)).max(1.0);
            let w_lim = (mw.rotational_mdot_kh_fac * m1 / t_kh)
                .min(base_w * mw.rotational_mdot_boost_fac);
            let mut w_rot = if base_w * enh < w_lim {
                base_w * enh
            } else {
                enh = w_lim / base_w.max(1.0e-30);
                w_lim
            };
            let prev_w = -model.aux.step_hist.mstar_dot_old;
            if w_rot > 0.0
                && prev_w > 0.0
                && mw.max_mdot_jump_for_rotation > 0.0
            {
                w_rot = w_rot.min(mw.max_mdot_jump_for_rotation * prev_w);
            }
            if mw.max_wind > 0.0 {
                w_rot = w_rot.min(mw.max_wind * M_SUN / SECYER);
            }
            rot_boost = enh;
            mdot_out = -w_rot;
        }
    }

    model.mstar_dot = mdot_out;
    model.aux.step_hist.rotational_mdot_boost_old = rot_boost;
    Ok(mdot_out)
}
