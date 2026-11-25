// SPDX-License-Identifier: GPL-3.0-or-later
//! `update_rotation_mixing_info` (`star/private/mix_info.f90:1561-1757`),
//! which combines the coefficients of `set_rotation_mixing_info`
//! ([`crate::rotation_mix_info`]) into `D_mix` and the angular-momentum
//! viscosities.

use flexmesa_const::MixingType;

use crate::controls::StarError;
use crate::model::StarModel;
use crate::rotation::conserve_shell_j_rot_and_distortion;

pub(crate) fn eval_d_mix_non_rotation_vec(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    if model.aux.mixing.have_mixing_info
        && model.aux.mixing.d_mix_non_rotation.len() == nz
    {
        return model.aux.mixing.d_mix_non_rotation.clone();
    }
    let mw = &model.controls.mix_wind;
    if !model.controls.rotation_flag
        || (mw.am_d_mix_factor <= 0.0 && mw.d_visc_factor <= 0.0)
    {
        return (0..nz)
            .map(|k| match model.mixing_type.get(k).copied() {
                Some(MixingType::NoMixing | MixingType::RotationMixing)
                | None => 0.0,
                Some(_) => model.d_mix.get(k).copied().unwrap_or(0.0),
            })
            .collect();
    }
    let d_rot = &model.aux.rotmix.d_mix_rotation;
    (0..nz)
        .map(|k| match model.mixing_type.get(k).copied() {
            Some(MixingType::NoMixing | MixingType::RotationMixing) | None => {
                0.0
            }
            Some(_) => (model.d_mix.get(k).copied().unwrap_or(0.0)
                - d_rot.get(k).copied().unwrap_or(0.0))
            .max(0.0),
        })
        .collect()
}

/// `update_rotation_mixing_info` (`star/private/mix_info.f90:1561-1757`):
/// `set_rotation_mixing_info`, the rotational part of `D_mix`, and the
/// angular-momentum viscosities `am_nu_non_rot`, `am_nu_rot`, `am_nu_omega`
/// and `am_nu_j`. Returns `D_mix_rotation` (face 0 is not set upstream and
/// is returned as 0).
///
/// The `D_omega_flag` and `am_nu_rot_flag` branches (`:1622-1624`,
/// `:1648-1667`, `:1685-1702`, `set_am_nu_rot` at `:1796-1954`) are ported.
/// `check_D_omega` (`:1594`) is not: `set_rotation_mixing_info` already
/// returns an error for any bad `D_omega`, so it can never fire.
///
/// # Errors
/// Propagates [`crate::rotation_mix_info::set_rotation_mixing_info`] errors
/// and returns [`StarError::ConvergenceFailure`] for a negative `am_nu_omega`
/// or `am_nu_j` (`:1706-1724`).
pub fn update_rotation_mixing_info(
    model: &mut StarModel,
) -> Result<Vec<f64>, StarError> {
    let nz = model.nz;
    model.aux.am_nu_non_rot.resize(nz, 0.0);
    model.aux.am_nu_rot.resize(nz, 0.0);
    model.aux.am_nu_omega.resize(nz, 0.0);
    model.aux.am_nu_j.resize(nz, 0.0);
    if !model.controls.rotation_flag || nz == 0 {
        return Ok(vec![0.0; nz]);
    }
    if model.aux.mixing.d_mix_non_rotation.len() != nz {
        model.aux.mixing.d_mix_non_rotation =
            eval_d_mix_non_rotation_vec(model);
    }

    crate::rotation_mix_info::set_rotation_mixing_info(model)?;

    let mw = model.controls.mix_wind.clone();
    // Rotation part for mixing abundances (`:1597-1643`).
    let full_on = mw.d_mix_rotation_max_logt_full_on;
    let full_off = mw.d_mix_rotation_min_logt_full_off;
    let full_on_tau = mw.d_mix_rotation_min_tau_full_on;
    let full_off_tau = mw.d_mix_rotation_min_tau_full_off;
    let mut d_mix_rotation = vec![0.0_f64; nz];
    {
        let rm = &model.aux.rotmix;
        for k in 1..nz {
            let lgt = model.ln_t[k] / flexmesa_const::LN10;
            // Upstream first sets `f` from `lgT` (`:1604-1611`) and then
            // unconditionally overwrites it from `tau`, whose middle case
            // reuses the `lgT` formula (`star/private/mix_info.f90:1614-1621`);
            // only the second assignment takes effect (deferred upstream bug #496).
            let f = if model.tau[k] >= full_on_tau {
                1.0
            } else if model.tau[k] <= full_off_tau {
                0.0
            } else {
                (lgt - full_on) / (full_off - full_on)
            };
            d_mix_rotation[k] = if model.controls.d_omega_flag {
                f * mw.am_d_mix_factor * rm.d_omega[k]
            } else {
                f * mw.am_d_mix_factor
                    * (mw.d_dsi_factor * rm.d_dsi[k]
                        + mw.d_sh_factor * rm.d_sh[k]
                        + mw.d_ssi_factor * rm.d_ssi[k]
                        + mw.d_es_factor * rm.d_es[k]
                        + mw.d_gsf_factor * rm.d_gsf[k]
                        + mw.d_st_factor * rm.d_st[k])
            };
            model.d_mix[k] =
                model.aux.mixing.d_mix_non_rotation[k] + d_mix_rotation[k];
        }
    }
    model.aux.rotmix.d_mix_rotation.resize(nz, 0.0);
    model.aux.rotmix.d_mix_rotation[0] = 0.0;
    model.aux.rotmix.d_mix_rotation[1..nz]
        .copy_from_slice(&d_mix_rotation[1..nz]);

    // `:1648-1667`: without `am_nu_rot_flag`, `D_omega_flag` with
    // `use_D_omega_for_am_nu_rot = .false.` needs every non-negative
    // `am_nu_*_factor` to equal its `D_*_factor`.
    if !model.controls.am_nu_rot_flag
        && model.controls.d_omega_flag
        && !model.job.use_d_omega_for_am_nu_rot
    {
        let pairs = [
            (mw.am_nu_dsi_factor, mw.d_dsi_factor),
            (mw.am_nu_sh_factor, mw.d_sh_factor),
            (mw.am_nu_ssi_factor, mw.d_ssi_factor),
            (mw.am_nu_es_factor, mw.d_es_factor),
            (mw.am_nu_gsf_factor, mw.d_gsf_factor),
            (mw.am_nu_st_factor, mw.d_st_factor),
        ];
        if pairs.iter().any(|&(a, d)| a >= 0.0 && a != d) {
            return Err(StarError::InvalidInput(String::from(
                "update_rotation_mixing_info: an am_nu factor >= 0 differs from the corresponding D_omega factor; set am_nu_rot_flag for smoothing like D_omega (star/private/mix_info.f90:1648-1667)",
            )));
        }
    }

    // `If am_nu_..._factor < -1, use the D_..._factor` (`:1662-1669`).
    let neg_or = |a: f64, d: f64| if a < 0.0 { d } else { a };
    let am_nu_dsi = neg_or(mw.am_nu_dsi_factor, mw.d_dsi_factor);
    let am_nu_sh = neg_or(mw.am_nu_sh_factor, mw.d_sh_factor);
    let am_nu_ssi = neg_or(mw.am_nu_ssi_factor, mw.d_ssi_factor);
    let am_nu_es = neg_or(mw.am_nu_es_factor, mw.d_es_factor);
    let am_nu_gsf = neg_or(mw.am_nu_gsf_factor, mw.d_gsf_factor);
    let am_nu_st = neg_or(mw.am_nu_st_factor, mw.d_st_factor);
    let am_nu_visc = neg_or(mw.am_nu_visc_factor, mw.d_visc_factor);

    // `:1673-1676`.
    let ye_center = model.ye.get(nz - 1).copied().unwrap_or(0.0);
    let set_min_am_nu_non_rot = mw.set_min_am_nu_non_rot
        && ye_center >= mw.min_center_ye_for_min_am_nu_non_rot
        && !mw.set_uniform_am_nu_non_rot;

    // `:1685-1688`.
    if model.controls.am_nu_rot_flag {
        set_am_nu_rot(
            model,
            [
                am_nu_visc, am_nu_dsi, am_nu_sh, am_nu_ssi, am_nu_es,
                am_nu_gsf, am_nu_st,
            ],
        )?;
    }

    // `:1690-1724`.
    let rm = &model.aux.rotmix;
    for k in 0..nz {
        model.aux.am_nu_non_rot[k] = if mw.set_uniform_am_nu_non_rot {
            mw.uniform_am_nu_non_rot
        } else {
            mw.am_nu_factor
                * mw.am_nu_non_rotation_factor
                * model.aux.mixing.d_mix_non_rotation[k]
        };
        if set_min_am_nu_non_rot {
            model.aux.am_nu_non_rot[k] =
                mw.min_am_nu_non_rot.max(model.aux.am_nu_non_rot[k]);
        }
        if model.controls.am_nu_rot_flag {
            // Already set by `set_am_nu_rot` above.
        } else if model.controls.d_omega_flag
            && model.job.use_d_omega_for_am_nu_rot
        {
            model.aux.am_nu_rot[k] = mw.am_nu_factor * rm.d_omega[k];
        } else {
            model.aux.am_nu_rot[k] = mw.am_nu_factor
                * (am_nu_visc * rm.d_visc[k]
                    + am_nu_dsi * rm.d_dsi[k]
                    + am_nu_sh * rm.d_sh[k]
                    + am_nu_ssi * rm.d_ssi[k]
                    + am_nu_es * rm.d_es[k]
                    + am_nu_gsf * rm.d_gsf[k]
                    + am_nu_st * rm.nu_st[k]);
        }
        if model.aux.am_nu_rot[k] < 0.0 {
            model.aux.am_nu_rot[k] = 0.0;
        }
        model.aux.am_nu_omega[k] = mw.am_nu_omega_non_rot_factor
            * model.aux.am_nu_non_rot[k]
            + mw.am_nu_omega_rot_factor * model.aux.am_nu_rot[k];
        if model.aux.am_nu_omega[k] < 0.0 {
            return Err(StarError::ConvergenceFailure(format!(
                "update_rotation_mixing_info: am_nu_omega({}) < 0: {:e} (star/private/mix_info.f90:1706-1714)",
                k + 1,
                model.aux.am_nu_omega[k]
            )));
        }
        model.aux.am_nu_j[k] = mw.am_nu_j_non_rot_factor
            * model.aux.am_nu_non_rot[k]
            + mw.am_nu_j_rot_factor * model.aux.am_nu_rot[k];
        if model.aux.am_nu_j[k] < 0.0 {
            return Err(StarError::ConvergenceFailure(format!(
                "update_rotation_mixing_info: am_nu_j({}) < 0: {:e} (star/private/mix_info.f90:1716-1724)",
                k + 1,
                model.aux.am_nu_j[k]
            )));
        }
    }

    Ok(d_mix_rotation)
}

/// `set_am_nu_rot` (`star/private/mix_info.f90:1796-1954`) with the
/// effective factors `[visc, DSI, SH, SSI, ES, GSF, ST]` of the caller.
fn set_am_nu_rot(
    model: &mut StarModel,
    fac: [f64; 7],
) -> Result<(), StarError> {
    let nz = model.nz;
    let mw = &model.controls.mix_wind;
    let rm = &model.aux.rotmix;
    let am = &mut model.aux.am_nu_rot;
    let bad = |what: &str, k: usize, v: f64| {
        StarError::ConvergenceFailure(format!(
            "set_am_nu_rot: {what}({}) is bad: {v:e} (star/private/mix_info.f90:1830-1839, 1942-1952)",
            k + 1
        ))
    };
    if model.aux.doing_finish_load_model {
        am[..nz].fill(0.0);
    } else {
        let [visc, dsi, sh, ssi, es, gsf, st] = fac;
        for k in 0..nz {
            if model.q[k] <= mw.max_q_for_nu_omega_zero_in_convection_region
                && model.mixing_type[k] == MixingType::ConvectiveMixing
            {
                am[k] = 0.0;
                continue;
            }
            let am_nu_rot_source = mw.am_nu_factor
                * (visc * rm.d_visc[k]
                    + dsi * rm.d_dsi[k]
                    + sh * rm.d_sh[k]
                    + ssi * rm.d_ssi[k]
                    + es * rm.d_es[k]
                    + gsf * rm.d_gsf[k]
                    + st * rm.nu_st[k]);
            if !am_nu_rot_source.is_finite() {
                return Err(bad("am_nu_rot_source", k, am_nu_rot_source));
            }
            am[k] = am_nu_rot_source;
        }
        if mw.smooth_am_nu_rot > 0 {
            crate::rotation_mix_info::smooth_for_rotation(
                &mut am[..nz],
                mw.smooth_am_nu_rot,
            );
        }
        if mw.nu_omega_mixing_rate > 0.0 && model.dt > 0.0 {
            crate::rotation_mix_info::mix_rotational_coefficient(
                &mut am[..nz],
                &model.mixing_type,
                mw.nu_omega_mixing_rate,
                mw.nu_omega_mixing_across_convection_boundary,
                model.dt,
                crate::rotation_mix_info::MixVariant::AmNuRot,
            )?;
        }
    }
    for k in 0..nz {
        if !am[k].is_finite() {
            return Err(bad("am_nu_rot before return", k, am[k]));
        }
        if am[k] < 0.0 {
            am[k] = 0.0;
        }
    }
    Ok(())
}

/// Adds the rotational part to `D_mix` via [`update_rotation_mixing_info`]
/// and marks faces where it dominates as `RotationMixing`.
///
/// # Errors
/// Propagates [`update_rotation_mixing_info`] errors.
pub(crate) fn augment_d_mix_with_rotation(
    model: &mut StarModel,
) -> Result<(), StarError> {
    let nz = model.nz;
    model.aux.mixing.d_mix_non_rotation.resize(nz, 0.0);
    model
        .aux
        .mixing
        .d_mix_non_rotation
        .copy_from_slice(&model.d_mix[..nz]);
    let d_mix_rotation = update_rotation_mixing_info(model)?;
    for k in 0..nz {
        if model.d_mix[k] != 0.0
            && model.aux.mixing.d_mix_non_rotation[k] < d_mix_rotation[k]
        {
            model.mixing_type[k] = MixingType::RotationMixing;
        }
    }
    Ok(())
}

/// Restores shell `j_rot` and the distortion factors and, for `dt > 0`,
/// runs `solve_omega_mix` over `dt` (half of it with `premix_omega`).
///
/// # Errors
/// Propagates [`update_rotation_mixing_info`] and `do_solve_omega_mix`
/// errors, and returns [`StarError::ConvergenceFailure`] when
/// `do_solve_omega_mix` asks for a retry or termination
/// (`star/private/solve_omega_mix.f90:136, 153`), since no caller of this
/// helper can redo the step.
pub(crate) fn sync_post_henyey_rotation(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 || model.j_rot.len() != nz {
        return Ok(());
    }
    conserve_shell_j_rot_and_distortion(model);
    if dt <= 0.0 || nz < 2 {
        return Ok(());
    }
    let mw = &model.controls.mix_wind;
    let d_mix_changed = model.aux.mixing.d_mix_non_rotation.len() != nz
        || (mw.am_d_mix_factor <= 0.0
            && mw.d_visc_factor <= 0.0
            && model.d_mix[..nz] != model.aux.mixing.d_mix_non_rotation[..nz]);
    let nu_unpopulated = model.aux.am_nu_omega.len() != nz
        || (model.aux.am_nu_omega.iter().all(|&v| v == 0.0)
            && model.aux.am_nu_j.iter().all(|&v| v == 0.0)
            && model.d_mix.iter().any(|&v| v > 0.0));
    if d_mix_changed || nu_unpopulated {
        model.aux.mixing.d_mix_non_rotation.resize(nz, 0.0);
        model
            .aux
            .mixing
            .d_mix_non_rotation
            .copy_from_slice(&model.d_mix[..nz]);
        update_rotation_mixing_info(model)?;
    }
    let dt_tr = if model.controls.mix_wind.premix_omega {
        0.5 * dt
    } else {
        dt
    };
    match crate::omega_mix::do_solve_omega_mix(model, dt_tr)? {
        crate::omega_mix::OmegaMixOutcome::KeepGoing => {}
        crate::omega_mix::OmegaMixOutcome::Retry(msg)
        | crate::omega_mix::OmegaMixOutcome::Terminate(msg) => {
            return Err(StarError::ConvergenceFailure(format!(
                "sync_post_henyey_rotation: solve_omega_mix failed: {msg}"
            )));
        }
    }
    crate::hydro_rotation::set_rotation_info(model, false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use flexmesa_const::MixingType;

    use super::{eval_d_mix_non_rotation_vec, update_rotation_mixing_info};
    use crate::controls::{StarControls, StarJobControls};
    use crate::zams::load_zams_model;

    fn rotating_zams_1m() -> crate::model::StarModel {
        let mut ctrl = StarControls {
            initial_mass: 1.0,
            zbase: 0.02,
            rotation_flag: true,
            ..StarControls::default()
        };
        ctrl.mix_wind.am_d_mix_factor = 1.0 / 30.0;
        ctrl.mix_wind.d_dsi_factor = 0.0;
        ctrl.mix_wind.d_sh_factor = 1.0;
        ctrl.mix_wind.d_ssi_factor = 1.0;
        ctrl.mix_wind.d_es_factor = 1.0;
        ctrl.mix_wind.d_gsf_factor = 1.0;
        ctrl.mix_wind.d_st_factor = 1.0;
        let mut model =
            load_zams_model(ctrl, StarJobControls::default()).expect("1M ZAMS");
        model.omega.fill(2.0e-5);
        crate::rotation::update_rotation_distortion_factors(&mut model);
        crate::hydro_vars::set_hydro_vars(&mut model, false, false)
            .expect("set_hydro_vars");
        model
    }

    #[test]
    fn test_eval_d_mix_non_rotation_vec_reconstructs_when_cache_missing() {
        let mut m = rotating_zams_1m();
        let nz = m.nz;
        assert!(nz > 4);
        m.aux.mixing.have_mixing_info = false;
        m.aux.mixing.d_mix_non_rotation.clear();
        m.aux.rotmix.d_mix_rotation.resize(nz, 0.0);
        m.aux.rotmix.d_mix_rotation[1] = 250.0;
        m.aux.rotmix.d_mix_rotation[2] = 500.0;
        m.mixing_type[1] = MixingType::RotationMixing;
        m.d_mix[1] = 250.0;
        m.mixing_type[2] = MixingType::ConvectiveMixing;
        m.d_mix[2] = 1.5e12 + 500.0;
        m.mixing_type[3] = MixingType::NoMixing;
        m.d_mix[3] = 999.0;

        let got = eval_d_mix_non_rotation_vec(&m);
        assert_eq!(got.len(), nz);
        assert_eq!(got[1], 0.0);
        assert!((got[2] - 1.5e12).abs() <= 1.0e-3);
        assert_eq!(got[3], 0.0);
    }

    #[test]
    fn test_update_rotation_mixing_info_clears_face0_and_handles_flags() {
        let mut m = rotating_zams_1m();
        let nz = m.nz;
        m.aux.rotmix.d_mix_rotation.resize(nz, 1234.0);
        m.controls.d_omega_flag = false;
        m.controls.am_nu_rot_flag = true;
        m.controls.mix_wind.smooth_am_nu_rot = 2;
        m.controls.mix_wind.nu_omega_mixing_rate = 1.0;
        m.dt = 1.0e10;

        let d_rot =
            update_rotation_mixing_info(&mut m).expect("am_nu_rot_flag update");
        assert_eq!(d_rot[0], 0.0);
        assert_eq!(m.aux.rotmix.d_mix_rotation[0], 0.0);
        assert_eq!(m.aux.am_nu_rot[0], 0.0);

        // `doing_finish_load_model` zeroes `am_nu_rot` (`mix_info.f90:1810`).
        m.aux.doing_finish_load_model = true;
        update_rotation_mixing_info(&mut m).expect("finish load model zeros");
        assert!(m.aux.am_nu_rot[..nz].iter().all(|&v| v == 0.0));
    }
}
