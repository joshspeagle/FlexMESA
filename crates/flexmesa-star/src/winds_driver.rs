// SPDX-License-Identifier: GPL-3.0-or-later
//! Control-driven stellar wind and mass-change driver (`set_mdot`,
//! `do_set_mdot`, `eval_super_eddington_wind`, `eval_rlo_wind`) matching
//! `star/private/winds.f90:54-476, 637-703`.

use flexmesa_const::{
    CLIGHT, CRAD, LN10, L_SUN, M_SUN, PI, R_SUN, SECYER, STANDARD_CGRAV,
};

use crate::controls::StarError;
use crate::core_boundaries::compute_core_masses;
use crate::inlist_runner::{DefaultStarExtras, StarExtras};
use crate::model::StarModel;
use crate::report_utils::{
    eval_ledd_erg_s, eval_photosphere_info, eval_surf_avg_rotation_info,
};
use crate::winds::{eval_scheme_rate, DutchLowTScheme, WindScheme};

#[inline]
fn surface_teff(model: &StarModel, l_surf: f64, r_surf: f64) -> f64 {
    if model.teff > 0.0 {
        model.teff
    } else {
        flexmesa_math::pow(
            l_surf / (PI * r_surf * r_surf * CRAD * CLIGHT),
            0.25,
        )
    }
}

/// Evaluates a single wind scheme and writes `model.mstar_dot` (`g s^-1`).
pub fn set_mdot(
    model: &mut StarModel,
    scheme: WindScheme,
    dutch_low_t: DutchLowTScheme,
    eta: f64,
) -> f64 {
    if model.nz == 0
        || matches!(scheme, WindScheme::None | WindScheme::Other)
        || eta <= 0.0
    {
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
        WindScheme::None | WindScheme::Other => 0.0,
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

fn net_h1_he4_surface_center(model: &StarModel) -> (f64, f64, f64, f64) {
    let nz = model.nz;
    let mut h1_idx = None;
    let mut he4_idx = None;
    for (j, name) in model.iso_names.iter().enumerate() {
        if name.eq_ignore_ascii_case("h1") {
            h1_idx = Some(j);
        } else if name.eq_ignore_ascii_case("he4") {
            he4_idx = Some(j);
        }
    }
    let (surf_h1, cen_h1) = if let Some(j) = h1_idx {
        (model.cell_xa(0)[j], model.cell_xa(nz - 1)[j])
    } else if model.species == 0 {
        (
            model.xh_mass.first().copied().unwrap_or(0.0),
            model.xh_mass.last().copied().unwrap_or(0.0),
        )
    } else {
        (0.0, 0.0)
    };
    let (surf_he4, cen_he4) = if let Some(j) = he4_idx {
        (model.cell_xa(0)[j], model.cell_xa(nz - 1)[j])
    } else if model.species == 0 {
        (
            model.yhe_mass.first().copied().unwrap_or(0.0),
            model.yhe_mass.last().copied().unwrap_or(0.0),
        )
    } else {
        (0.0, 0.0)
    };
    (surf_h1, cen_h1, surf_he4, cen_he4)
}

/// Evaluates the super-Eddington wind rate $\dot{M}_{\mathrm{sEdd}} \ge 0$ in
/// $\mathrm{g\,s^{-1}}$ matching `eval_super_eddington_wind` in
/// `star/private/winds.f90:637-658`.
#[must_use]
pub fn eval_super_eddington_wind(
    model: &StarModel,
    l1: f64,
    m1: f64,
    r1: f64,
    prev_ledd: f64,
) -> f64 {
    let c = &model.controls;
    if c.super_eddington_scaling_factor <= 0.0 {
        return 0.0;
    }
    let leff = l1 / c.mix_wind.super_eddington_wind_ledd_factor;
    if leff <= prev_ledd {
        return 0.0;
    }
    let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
    let vesc2 = 2.0 * cgrav * m1 / r1;
    c.super_eddington_scaling_factor * (leff - prev_ledd) / (0.5 * vesc2)
}

/// Evaluates single-star Roche-lobe overflow wind in $M_\odot\,\mathrm{yr}^{-1}$
/// and updates `xfer_ratio` matching `eval_rlo_wind` in
/// `star/private/winds.f90:661-703`.
#[must_use]
pub fn eval_rlo_wind(
    model: &StarModel,
    l_surf_lsun: f64,
    r_rsun: f64,
    teff: f64,
    xfer_ratio: &mut f64,
) -> f64 {
    let mw = &model.controls.mix_wind;
    if mw.rlo_scaling_factor <= 0.0 {
        return 0.0;
    }
    if l_surf_lsun < mw.rlo_wind_min_l {
        return 0.0;
    }
    if teff > mw.rlo_wind_max_teff {
        return 0.0;
    }
    let mut scale_height = mw.rlo_wind_scale_height;
    if scale_height <= 0.0 {
        let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);
        let p0 = model.peos.first().copied().unwrap_or(0.0);
        let m0 = model.m.first().copied().unwrap_or(model.mstar);
        let rho0 = model.rho.first().copied().unwrap_or(1.0);
        let r0 = model.r.first().copied().unwrap_or(1.0);
        scale_height = p0 / (cgrav * m0 * rho0 / (r0 * r0)) / R_SUN;
    }
    let roche_lobe_radius = mw.rlo_wind_roche_lobe_radius;
    let ratio = r_rsun / roche_lobe_radius;
    if ratio < 1.0 {
        if ratio < mw.roche_lobe_xfer_full_on {
            return 0.0;
        }
        if ratio > mw.roche_lobe_xfer_full_off {
            *xfer_ratio = 0.0;
            return 0.0;
        }
        let f = (mw.roche_lobe_xfer_full_off - ratio)
            / (mw.roche_lobe_xfer_full_off - mw.roche_lobe_xfer_full_on);
        *xfer_ratio = 0.5 * (1.0 - flexmesa_math::cospi(f));
        return 0.0;
    }
    let expo = (6.0 * LN10).min((r_rsun - roche_lobe_radius) / scale_height);
    let mdot = mw.rlo_wind_base_mdot * flexmesa_math::exp(expo);
    mw.rlo_scaling_factor * mdot
}

/// Evaluates the full control-driven mass-change rate `model.mstar_dot`
/// (`g s^-1`, negative for mass loss, positive for accretion) matching
/// `do_set_mdot` in `star/private/winds.f90:54-476`.
///
/// # Errors
/// Returns [`StarError`] if `hot_wind_full_on_t < cool_wind_full_on_t` or an
/// unknown wind scheme name is configured.
pub fn do_set_mdot(model: &mut StarModel) -> Result<f64, StarError> {
    do_set_mdot_with_extras(model, &DefaultStarExtras)
}

#[allow(clippy::too_many_arguments)]
fn eval_wind_for_scheme(
    model: &StarModel,
    extras: &dyn StarExtras,
    scheme_name: &str,
    l1: f64,
    m1: f64,
    r1: f64,
    t1: f64,
    surface_h1: f64,
    surface_he4: f64,
    z_surf: f64,
    cgrav: f64,
) -> Result<f64, StarError> {
    // Match `star/private/winds.f90:319-406`: `use_other` overrides scheme-name
    // matching so `use_other_wind = true` allows custom descriptive scheme strings
    // (such as `'Kudritzki'` or `'Grafener'`) without rejecting them.
    let use_other = model.controls.use_other_wind
        || scheme_name.trim().eq_ignore_ascii_case("other");
    let scheme = if use_other {
        WindScheme::Other
    } else {
        let parsed = WindScheme::parse_name(scheme_name).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "unknown name for wind scheme : {scheme_name}"
            ))
        })?;
        if parsed == WindScheme::None {
            return Ok(0.0);
        }
        parsed
    };
    let reimers_base = 4.0e-13 * (l1 * r1 / m1) / (L_SUN * R_SUN / M_SUN);
    if reimers_base <= 0.0 || !reimers_base.is_finite() {
        return Err(StarError::InvalidInput(format!(
            "bad value for wind: {reimers_base} (L1={l1}, R1={r1}, M1={m1})"
        )));
    }
    if use_other {
        return extras.other_wind(model, l1, m1, r1, t1)?.ok_or_else(|| {
            StarError::InvalidInput(String::from(
                "use_other_wind / scheme='other' requires an other_wind hook",
            ))
        });
    }
    // Validate `Dutch_wind_lowT_scheme` lazily only when `'Dutch'` is active
    // (`star/private/winds.f90:576-593`).
    let dutch_low = if scheme == WindScheme::Dutch {
        DutchLowTScheme::parse_name(
            &model.controls.mix_wind.dutch_wind_lowt_scheme,
        )
        .ok_or_else(|| {
            StarError::InvalidInput(format!(
                "unknown Dutch_wind_lowT_scheme '{}'",
                model.controls.mix_wind.dutch_wind_lowt_scheme
            ))
        })?
    } else {
        DutchLowTScheme::DeJager
    };
    Ok(eval_scheme_rate(
        scheme,
        dutch_low,
        l1,
        m1,
        r1,
        t1,
        surface_h1,
        surface_he4,
        z_surf,
        cgrav,
        scheme_eta(model, scheme),
    ))
}

/// Evaluates the full control-driven mass-change rate `model.mstar_dot`
/// (`g s^-1`) using `extras` for `use_other_wind` (`winds.f90:54-476`).
///
/// # Errors
/// Returns [`StarError`] if wind controls are invalid or a required
/// `other_wind` hook fails or is missing.
pub fn do_set_mdot_with_extras(
    model: &mut StarModel,
    extras: &dyn StarExtras,
) -> Result<f64, StarError> {
    let nz = model.nz;
    if nz == 0 {
        model.mstar_dot = 0.0;
        model.aux.step_hist.super_eddington_wind_mdot = 0.0;
        model.aux.step_hist.doing_rlo_wind = false;
        model.aux.step_hist.explicit_mstar_dot = 0.0;
        return Ok(0.0);
    }
    if model.controls.hot_wind_full_on_t < model.controls.cool_wind_full_on_t {
        return Err(StarError::InvalidInput(String::from(
            "hot_wind_full_on_T < cool_wind_full_on_T in do_set_mdot",
        )));
    }

    // Photosphere inputs matching `evolve.f90:427` (`L_phot*Lsun, mstar, Teff`).
    let phot = eval_photosphere_info(model);
    let l1 = if phot.l_lsun > 0.0 {
        phot.l_lsun * L_SUN
    } else {
        model.lum[0].max(1.0)
    };
    let m1 = if model.mstar > 0.0 {
        model.mstar
    } else {
        model.m_grav[0].max(1.0)
    };
    let r_fallback = if phot.r_cm > 0.0 {
        phot.r_cm
    } else {
        model.r[0].max(1.0)
    };
    let t1 = surface_teff(model, l1, r_fallback);
    let r1 = (l1 / (PI * CRAD * CLIGHT * t1 * t1 * t1 * t1)).sqrt();
    let cgrav = model.cgrav.first().copied().unwrap_or(STANDARD_CGRAV);

    let prev_ledd = model
        .aux
        .step_hist
        .prev_ledd
        .unwrap_or_else(|| eval_ledd_erg_s(model));
    let starting_t_center = model
        .aux
        .step_hist
        .starting_t_center
        .unwrap_or_else(|| model.t[nz - 1]);

    let (surface_h1, center_h1, surface_he4, center_he4) =
        net_h1_he4_surface_center(model);
    let z_surf = if model.kap_config.zbase >= 0.0 {
        model.kap_config.zbase
    } else {
        1.0 - (surface_h1 + surface_he4)
    };

    let mut wind_mdot = 0.0_f64;
    let mut using_wind_scheme = false;
    let mut xfer_ratio = 1.0_f64;

    // 1. Super-Eddington wind (`eval_super_eddington_wind`, `winds.f90:92-102, 637-658`).
    let sedd_mdot = eval_super_eddington_wind(model, l1, m1, r1, prev_ledd);
    model.aux.step_hist.super_eddington_wind_mdot = sedd_mdot;
    if sedd_mdot > wind_mdot {
        wind_mdot = sedd_mdot;
    }

    // 2. Single-star RLO wind (`eval_rlo_wind`, `winds.f90:104-116, 661-703`).
    let rlo_mdot =
        eval_rlo_wind(model, l1 / L_SUN, r1 / R_SUN, t1, &mut xfer_ratio)
            * M_SUN
            / SECYER;
    model.aux.step_hist.doing_rlo_wind = rlo_mdot != 0.0;
    if model.aux.step_hist.doing_rlo_wind && rlo_mdot > wind_mdot {
        wind_mdot = rlo_mdot;
    }

    // 3. Explicit `mass_change` (`M_sun yr^-1`, positive = accretion -> negative `wind_mdot`,
    //    `winds.f90:133-141`).
    let mass_change = model.controls.mass_change;
    if (mass_change > 0.0 && wind_mdot == 0.0)
        || (mass_change < 0.0 && -mass_change * M_SUN / SECYER > wind_mdot)
    {
        wind_mdot = -mass_change * M_SUN / SECYER;
        if mass_change > 0.0 && xfer_ratio < 1.0 {
            wind_mdot *= xfer_ratio;
        }
    }

    // 4. Separate hot and cool wind evaluations (`winds.f90:150-190, 314-409`).
    let hot_on = model.controls.hot_wind_full_on_t;
    let cool_on = model.controls.cool_wind_full_on_t;
    let hot_w = if t1 >= cool_on {
        eval_wind_for_scheme(
            model,
            extras,
            &model.controls.mix_wind.hot_wind_scheme,
            l1,
            m1,
            r1,
            t1,
            surface_h1,
            surface_he4,
            z_surf,
            cgrav,
        )?
    } else {
        0.0
    };
    let cool_w = if t1 <= hot_on {
        let cool_name = if center_h1 < 0.01
            && center_he4 < model.controls.mix_wind.rgb_to_agb_wind_switch
        {
            &model.controls.cool_wind_agb_scheme
        } else {
            &model.controls.cool_wind_rgb_scheme
        };
        eval_wind_for_scheme(
            model,
            extras,
            cool_name,
            l1,
            m1,
            r1,
            t1,
            surface_h1,
            surface_he4,
            z_surf,
            cgrav,
        )?
    } else {
        0.0
    };

    let wind_msun_yr = if t1 >= hot_on {
        hot_w
    } else if t1 <= cool_on {
        cool_w
    } else if hot_on == cool_on {
        0.5 * (hot_w + cool_w)
    } else {
        let divisor = hot_on - cool_on;
        let beta = ((hot_on - t1) / divisor).min(1.0);
        let alfa = 1.0 - beta;
        alfa * hot_w + beta * cool_w
    };

    let wind_cgs = wind_msun_yr * M_SUN / SECYER;
    if wind_cgs > wind_mdot.abs() {
        using_wind_scheme = true;
        wind_mdot = wind_cgs;
    }

    // 5. Wind-scheme-only gates (`no_wind_if_no_rotation`, timestep ramp; `winds.f90:194-214`).
    let mw = &model.controls.mix_wind;
    if using_wind_scheme {
        if mw.no_wind_if_no_rotation && !model.controls.rotation_flag {
            model.mstar_dot = 0.0;
            return Ok(0.0);
        }
        if model.dt > 0.0 && model.dt < mw.mass_change_full_on_dt {
            if model.dt <= mw.mass_change_full_off_dt {
                model.mstar_dot = 0.0;
                return Ok(0.0);
            }
            let alfa = (model.dt - mw.mass_change_full_off_dt)
                / (mw.mass_change_full_on_dt - mw.mass_change_full_off_dt);
            wind_mdot *= alfa;
        }
    }

    // 6. Super-Eddington wind boost (`winds.f90:216-236`).
    if wind_mdot >= 0.0
        && model.controls.super_eddington_scaling_factor <= 0.0
        && prev_ledd > 0.0
    {
        let l_div_ledd = l1 / prev_ledd;
        let full_off = mw.wind_boost_full_off_l_div_ledd;
        if l_div_ledd > full_off {
            let full_on = mw.wind_boost_full_on_l_div_ledd;
            let max_boost = mw.super_eddington_wind_max_boost;
            let boost = if l_div_ledd >= full_on {
                max_boost
            } else {
                1.0 + (max_boost - 1.0) * (l_div_ledd - full_off)
                    / (full_on - full_off)
            };
            wind_mdot *= boost;
        }
    }

    // 7. `min_wind` and `max_wind` (`winds.f90:240-252`).
    if wind_mdot >= 0.0 && mw.min_wind > 0.0 {
        let min_w = mw.min_wind * M_SUN / SECYER;
        if wind_mdot < min_w {
            wind_mdot = min_w;
        }
    }
    if wind_mdot >= 0.0 && mw.max_wind > 0.0 {
        let max_w = mw.max_wind * M_SUN / SECYER;
        if wind_mdot > max_w {
            wind_mdot = max_w;
        }
    }

    // 8. Central-temperature mass-loss taper (`winds.f90:255-268`).
    if wind_mdot >= 0.0 {
        if starting_t_center > mw.max_t_center_for_any_mass_loss {
            wind_mdot = 0.0;
        } else if starting_t_center > mw.max_t_center_for_full_mass_loss {
            wind_mdot *= (mw.max_t_center_for_any_mass_loss
                - starting_t_center)
                / (mw.max_t_center_for_any_mass_loss
                    - mw.max_t_center_for_full_mass_loss);
        }
    }

    // 9. Envelope-mass wind limits (`winds.f90:271-285`).
    if wind_mdot >= 0.0
        && (mw.wind_h_envelope_limit > 0.0
            || mw.wind_h_he_envelope_limit > 0.0
            || mw.wind_he_layer_limit > 0.0)
    {
        let cores = compute_core_masses(model);
        let star_mass = model.mstar / M_SUN;
        let h_env_mass = star_mass - cores.he_core_mass;
        let h_he_env_mass = star_mass - cores.co_core_mass;
        let he_layer_mass = cores.he_core_mass - cores.co_core_mass;
        if (mw.wind_h_envelope_limit > 0.0
            && h_env_mass < mw.wind_h_envelope_limit)
            || (mw.wind_h_he_envelope_limit > 0.0
                && h_he_env_mass < mw.wind_h_he_envelope_limit)
            || (mw.wind_he_layer_limit > 0.0
                && he_layer_mass < mw.wind_he_layer_limit)
        {
            wind_mdot = 0.0;
        }
    }

    model.mstar_dot = -wind_mdot;

    // 10. Heger, Langer & Woosley (2000) rotational enhancement (`winds.f90:289-302, 412-476`).
    let v_div_v_crit_avg_surf = if model.controls.rotation_flag {
        eval_surf_avg_rotation_info(model).v_div_v_crit
    } else {
        0.0
    };
    if model.mstar_dot < 0.0
        && (model.controls.mix_wind.min_wind > 0.0
            || using_wind_scheme
            || v_div_v_crit_avg_surf > 0.8)
    {
        apply_rotation_enhancement(
            model,
            cgrav,
            l1,
            m1,
            r1,
            v_div_v_crit_avg_surf,
        );
    }

    model.aux.step_hist.explicit_mstar_dot = model.mstar_dot;
    Ok(model.mstar_dot)
}

fn apply_rotation_enhancement(
    model: &mut StarModel,
    cgrav: f64,
    l1: f64,
    m1: f64,
    r1: f64,
    v_div_v_crit_avg_surf: f64,
) {
    let mw = &model.controls.mix_wind;
    if !model.controls.rotation_flag
        || mw.mdot_omega_power <= 0.0
        || model.mstar_dot >= 0.0
    {
        return;
    }

    let mut wind_mdot = -model.mstar_dot;
    let kh_timescale = 0.75 * cgrav * m1 * m1 / (r1 * l1);
    let wind_mdot_lim = (mw.rotational_mdot_kh_fac * m1 / kh_timescale)
        .min(wind_mdot * mw.rotational_mdot_boost_fac);

    let mut enhancement = flexmesa_math::pow(
        (1.0 - v_div_v_crit_avg_surf).max(1.0e-22),
        -mw.mdot_omega_power,
    );
    if mw.max_rotational_mdot_boost > 0.0
        && enhancement > mw.max_rotational_mdot_boost
    {
        enhancement = mw.max_rotational_mdot_boost;
    }

    if wind_mdot * enhancement < wind_mdot_lim {
        wind_mdot *= enhancement;
    } else {
        enhancement = wind_mdot_lim / wind_mdot;
        wind_mdot = wind_mdot_lim;
    }

    let wind_mdot_prev = -model.aux.step_hist.mstar_dot_old;
    if wind_mdot > 0.0 && wind_mdot_prev > 0.0 {
        wind_mdot =
            wind_mdot.min(mw.max_mdot_jump_for_rotation * wind_mdot_prev);
    }

    if wind_mdot >= 0.0 && mw.max_wind > 0.0 {
        let max_w = mw.max_wind * M_SUN / SECYER;
        if wind_mdot > max_w {
            wind_mdot = max_w;
        }
    }

    model.mstar_dot = -wind_mdot;
    // Note: upstream writes current-step `s% rotational_mdot_boost = enhancement`
    // (`winds.f90:474`), reset to `0d0` at `do_step_part1` (`evolve.f90:316`);
    // FlexMESA stores it in `rotational_mdot_boost_old` for history/retry callers.
    model.aux.step_hist.rotational_mdot_boost_old = enhancement;
}
