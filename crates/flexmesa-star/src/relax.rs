// SPDX-License-Identifier: GPL-3.0-or-later
//! Model changes applied while a run starts (`star/job/run_star_support.f90`):
//! `set_uniform_initial_composition`, uniform rotation (`star_set_uniform_omega`)
//! and core excision by temperature. The relaxes that evolve the model are in
//! `crate::relax_driver` and `crate::relax_mass_comp`.

use crate::controls::{StarError, StarJobControls};
use crate::model::StarModel;
pub use crate::relax_comp::{
    cell_center_xq, parse_angular_momentum_profile_str,
    parse_entropy_profile_str, parse_entropy_profile_with_eos_str,
    EntropyProfileSource, RelaxScalarInterpolant, ScalarProfile,
};
pub use crate::relax_mass_comp::{
    do_relax_angular_momentum, RelaxAngularMomentumConfig,
};

/// `set_uniform_initial_composition` (`star/job/run_star_support.f90:2198-2234`):
/// replaces every zone's mass fractions with the uniform mixture that
/// `star_set_standard_composition` builds from `initial_h1`, `initial_h2`,
/// `initial_he3`, `initial_he4` and `initial_zfracs`
/// (`star/private/adjust_xyz.f90:891-907`, `get_xa_for_standard_metals` at
/// `adjust_xyz.f90:991-1093`, `set_composition` at `adjust_xyz.f90:849-888`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when `get_xa_for_standard_metals`
/// fails or the mass fractions do not add to 1 within `1e-6`
/// (`adjust_xyz.f90:871-876`).
pub fn set_uniform_initial_composition(
    model: &mut StarModel,
    job: &StarJobControls,
) -> Result<(), StarError> {
    let xa = crate::accretion_xa::standard_metals_xa(
        &model.chem_id,
        &model.controls.mix_wind,
        job.initial_h1,
        job.initial_h2,
        job.initial_he3,
        job.initial_he4,
        job.initial_zfracs,
        job.dump_missing_metals_into_heaviest,
    )
    .map_err(|e| {
        StarError::InvalidInput(format!("set_uniform_initial_composition: {e}"))
    })?;
    let species = model.species;
    if xa.len() != species {
        return Err(StarError::InvalidInput(String::from(
            "set_composition requires number of species to match current model",
        )));
    }
    if (1.0 - xa.iter().sum::<f64>()).abs() > 1.0e-6 {
        return Err(StarError::InvalidInput(String::from(
            "set_composition requires new mass fractions to add to 1",
        )));
    }
    for k in 0..model.nz {
        model.cell_xa_mut(k).copy_from_slice(&xa);
    }
    Ok(())
}

/// Applies solid-body rotation $\Omega_k = \Omega_0$, computes shell-corrected
/// $i_{\mathrm{rot},k}$ and $j_{\mathrm{rot},k} = i_{\mathrm{rot},k}\Omega_0$,
/// and updates Roche distortion factors (`set_uniform_omega` in
/// `star/private/hydro_rotation.f90:501-543`).
pub fn apply_uniform_rotation(model: &mut StarModel, omega_0: f64) {
    model.controls.rotation_flag = true;
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    if model.omega.len() != nz {
        model.omega.resize(nz, 0.0);
    }
    if model.j_rot.len() != nz {
        model.j_rot.resize(nz, 0.0);
    }
    model.omega[..nz].fill(omega_0);
    // `fp_rot = ft_rot = 0` (`hydro_rotation.f90:503-507`).
    for v in [&mut model.aux.fp_rot, &mut model.aux.ft_rot] {
        if v.len() == nz {
            v[..nz].fill(0.0);
        }
    }
    crate::hydro_rotation::use_xh_to_update_i_rot_and_j_rot(model);
    // `hydro_rotation.f90:509-531` zeroes the rotational distortion and the
    // angular-momentum mixing arrays. Of those FlexMESA stores
    // `w_div_w_crit_roche` (`w_div_wc`, set again by `set_rotation_info`), the
    // `am_nu_*`/`am_sig_*` viscosities and `D_mix_non_rotation`; `r_polar`,
    // `r_equatorial`, `domega_dlnR`, `richardson_number`, `D_visc`, `D_DSI`,
    // `D_SH`, `D_SSI`, `D_ES`, `D_GSF`, `D_ST`, `nu_ST`, `omega_shear` and
    // `dynamo_B_*` are recomputed where they are used, not stored.
    let aux = &mut model.aux;
    for v in [
        &mut aux.w_div_wc,
        &mut aux.am_nu_rot,
        &mut aux.am_nu_non_rot,
        &mut aux.am_nu_omega,
        &mut aux.am_nu_j,
        &mut aux.am_sig_omega,
        &mut aux.am_sig_j,
        &mut aux.mixing.d_mix_non_rotation,
    ] {
        v.fill(0.0);
    }
    crate::hydro_rotation::set_rotation_info(model, false);
}

/// Excises all interior zones hotter than `t_cut_k` (`K`) and replaces the
/// interior with a rigid radiative/hydrostatic inner boundary `(M_center, R_center, L_center)`
/// (`remove_initial_center_by_temperature` in `star/private/relax.f90`).
///
/// # Errors
/// Returns [`StarError`] if the mesh cannot be normalized and repacked before
/// or after the excision.
pub fn excise_core_by_temperature(
    model: &mut StarModel,
    t_cut_k: f64,
) -> Result<(), StarError> {
    if t_cut_k <= 0.0 || model.nz < 32 {
        return Ok(());
    }
    let ln_t_cut = flexmesa_math::log(t_cut_k);
    let Some(raw_cut) = (16..model.nz).find(|&k| model.ln_t[k] >= ln_t_cut)
    else {
        return Ok(());
    };
    let k_cut = raw_cut.min(model.nz - 4);
    // Not an upstream port (#154); keeps the normalization this stand-in had
    // through sync_mesh_and_pack before #381.
    model.normalize_dqs()?;
    model.sync_mesh_and_pack()?;

    let m_c = model.m[k_cut];
    let r_c = model.r[k_cut];
    let l_c = model.lum[k_cut].max(1.0);
    if m_c <= 0.0 || m_c >= 0.999999 * model.mstar || r_c <= 0.0 {
        return Ok(());
    }

    let species = model.species;
    let mut trimmed = StarModel::new(
        k_cut,
        model.chem_id.clone(),
        model.job.clone(),
        model.controls.clone(),
    );
    trimmed.eos_config = model.eos_config;
    trimmed.kap_config = model.kap_config.clone();
    trimmed.model_number = model.model_number;
    trimmed.star_age = model.star_age;
    trimmed.star_age_old = model.star_age_old;
    trimmed.dt = model.dt;
    trimmed.dt_next = model.dt_next;
    trimmed.dt_old = model.dt_old;
    trimmed.mstar = model.mstar;
    trimmed.mstar_start = model.mstar;
    trimmed.mstar_old = model.mstar;
    trimmed.m_center = m_c;
    trimmed.r_center = r_c;
    trimmed.l_center = l_c;
    trimmed.xmstar = (model.mstar - m_c).max(1.0);
    trimmed.teff = model.teff;
    trimmed.tau_base = model.tau_base;

    trimmed.lnd.copy_from_slice(&model.lnd[..k_cut]);
    trimmed.ln_t.copy_from_slice(&model.ln_t[..k_cut]);
    trimmed.ln_r.copy_from_slice(&model.ln_r[..k_cut]);
    trimmed.lum.copy_from_slice(&model.lum[..k_cut]);
    if model.v.len() >= k_cut {
        trimmed.v.copy_from_slice(&model.v[..k_cut]);
    }
    if model.u.len() >= k_cut {
        trimmed.u.copy_from_slice(&model.u[..k_cut]);
    }
    if model.aux.alpha_rti.len() >= k_cut
        && trimmed.aux.alpha_rti.len() == k_cut
    {
        trimmed
            .aux
            .alpha_rti
            .copy_from_slice(&model.aux.alpha_rti[..k_cut]);
    }
    trimmed.dq.copy_from_slice(&model.dq[..k_cut]);
    trimmed.xa.copy_from_slice(&model.xa[..k_cut * species]);
    trimmed.cgrav.copy_from_slice(&model.cgrav[..k_cut]);
    if model.omega.len() >= k_cut && trimmed.omega.len() == k_cut {
        trimmed.omega.copy_from_slice(&model.omega[..k_cut]);
    }
    if model.j_rot.len() >= k_cut && trimmed.j_rot.len() == k_cut {
        trimmed.j_rot.copy_from_slice(&model.j_rot[..k_cut]);
    }

    let dq_sum: f64 = trimmed.dq.iter().sum();
    if dq_sum > 0.0 {
        for dq_k in &mut trimmed.dq {
            *dq_k /= dq_sum;
        }
    }
    trimmed.sync_mesh_and_pack()?;
    *model = trimmed;
    Ok(())
}
