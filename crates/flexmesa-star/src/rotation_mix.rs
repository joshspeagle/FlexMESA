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
            Some(_) => {
                (model.d_mix[k] - d_rot.get(k).copied().unwrap_or(0.0)).max(0.0)
            }
        })
        .collect()
}

/// `update_rotation_mixing_info` (`star/private/mix_info.f90:1561-1757`):
/// `set_rotation_mixing_info`, the rotational part of `D_mix`, and the
/// angular-momentum viscosities `am_nu_non_rot`, `am_nu_rot`, `am_nu_omega`
/// and `am_nu_j`. Returns `D_mix_rotation` (face 0 is not set upstream and
/// is returned as 0).
///
/// `D_omega_flag` and `am_nu_rot_flag` are not supported controls, so their
/// branches (`:1594, 1621-1626, 1668-1687, 1696-1699`) do not arise.
///
/// # Errors
/// Propagates [`crate::rotation_mix_info::set_rotation_mixing_info`] errors
/// and returns [`StarError::ConvergenceFailure`] for a negative `am_nu_omega`
/// or `am_nu_j` (`:1706-1724`).
pub(crate) fn update_rotation_mixing_info(
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
            d_mix_rotation[k] = f
                * mw.am_d_mix_factor
                * (mw.d_dsi_factor * rm.d_dsi[k]
                    + mw.d_sh_factor * rm.d_sh[k]
                    + mw.d_ssi_factor * rm.d_ssi[k]
                    + mw.d_es_factor * rm.d_es[k]
                    + mw.d_gsf_factor * rm.d_gsf[k]
                    + mw.d_st_factor * rm.d_st[k]);
            model.d_mix[k] =
                model.aux.mixing.d_mix_non_rotation[k] + d_mix_rotation[k];
        }
    }
    model.aux.rotmix.d_mix_rotation.resize(nz, 0.0);
    model.aux.rotmix.d_mix_rotation[1..nz]
        .copy_from_slice(&d_mix_rotation[1..nz]);

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
    let set_min_am_nu_non_rot = mw.set_min_am_nu_non_rot
        && model.ye[nz - 1] >= mw.min_center_ye_for_min_am_nu_non_rot
        && !mw.set_uniform_am_nu_non_rot;

    // `:1683-1724`.
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
        model.aux.am_nu_rot[k] = mw.am_nu_factor
            * (am_nu_visc * rm.d_visc[k]
                + am_nu_dsi * rm.d_dsi[k]
                + am_nu_sh * rm.d_sh[k]
                + am_nu_ssi * rm.d_ssi[k]
                + am_nu_es * rm.d_es[k]
                + am_nu_gsf * rm.d_gsf[k]
                + am_nu_st * rm.nu_st[k]);
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

pub(crate) fn augment_d_mix_with_rotation(model: &mut StarModel) {
    let nz = model.nz;
    model.aux.mixing.d_mix_non_rotation.resize(nz, 0.0);
    model
        .aux
        .mixing
        .d_mix_non_rotation
        .copy_from_slice(&model.d_mix[..nz]);
    let Ok(d_mix_rotation) = update_rotation_mixing_info(model) else {
        return;
    };
    for k in 0..nz {
        if model.d_mix[k] != 0.0
            && model.aux.mixing.d_mix_non_rotation[k] < d_mix_rotation[k]
        {
            model.mixing_type[k] = MixingType::RotationMixing;
        }
    }
}

pub(crate) fn sync_post_henyey_rotation(model: &mut StarModel, dt: f64) {
    let nz = model.nz;
    if !model.controls.rotation_flag || nz == 0 || model.j_rot.len() != nz {
        return;
    }
    conserve_shell_j_rot_and_distortion(model);
    if dt <= 0.0 || nz < 2 {
        return;
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
        let _ = update_rotation_mixing_info(model);
    }
    let dt_tr = if model.controls.mix_wind.premix_omega {
        0.5 * dt
    } else {
        dt
    };
    let _ = crate::omega_mix::do_solve_omega_mix(model, dt_tr);
    crate::hydro_rotation::set_rotation_info(model, false);
}
