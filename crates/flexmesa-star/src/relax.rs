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
/// `star/private/hydro_rotation.f90:492-535`).
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
    // `hydro_rotation.f90:509-531` zeroes `w_div_w_crit_roche`, `r_polar`,
    // `r_equatorial` (the latter two are not yet stored on `HydroAuxState`),
    // the angular-momentum mixing viscosities/sigmas, and the rotational mixing
    // arrays. Note `D_omega` is not zeroed here (`relax.f90:1307` zeroes it
    // when `do_relax_uniform_omega` finishes).
    let aux = &mut model.aux;
    for v in [
        &mut aux.w_div_wc,
        &mut aux.am_nu_rot,
        &mut aux.am_nu_non_rot,
        &mut aux.am_nu_omega,
        &mut aux.am_nu_j,
        &mut aux.am_sig_omega,
        &mut aux.am_sig_j,
        &mut aux.rotmix.domega_dlnr,
        &mut aux.rotmix.richardson_number,
        &mut aux.mixing.d_mix_non_rotation,
        &mut aux.rotmix.d_visc,
        &mut aux.rotmix.d_dsi,
        &mut aux.rotmix.d_sh,
        &mut aux.rotmix.d_ssi,
        &mut aux.rotmix.d_es,
        &mut aux.rotmix.d_gsf,
        &mut aux.rotmix.d_st,
        &mut aux.rotmix.nu_st,
        &mut aux.rotmix.omega_shear,
        &mut aux.rotmix.dynamo_b_r,
        &mut aux.rotmix.dynamo_b_phi,
    ] {
        v.resize(nz, 0.0);
        v.fill(0.0);
    }
    crate::hydro_rotation::set_rotation_info(model, false);
    // `s% need_to_setvars = .true.` (`hydro_rotation.f90:534`).
    model.aux.need_to_setvars = true;
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

#[cfg(test)]
mod tests {
    use super::apply_uniform_rotation;
    use crate::controls::{StarControls, StarJobControls};
    use crate::zams::load_zams_model;

    #[test]
    fn test_apply_uniform_rotation_sets_need_to_setvars_and_zeroes_rotmix() {
        let mut model = load_zams_model(
            StarControls::default(),
            StarJobControls::default(),
        )
        .expect("1 Msun ZAMS");
        let nz = model.nz;

        // First call with default empty `RotMixArrays` resizes and zeroes all
        // 13 `rotmix` arrays and sets `need_to_setvars = true`.
        model.aux.need_to_setvars = false;
        apply_uniform_rotation(&mut model, 1.0e-5);
        assert!(model.aux.need_to_setvars);

        let rm = &model.aux.rotmix;
        for (name, arr) in [
            ("domega_dlnr", &rm.domega_dlnr),
            ("richardson_number", &rm.richardson_number),
            ("d_visc", &rm.d_visc),
            ("d_dsi", &rm.d_dsi),
            ("d_sh", &rm.d_sh),
            ("d_ssi", &rm.d_ssi),
            ("d_es", &rm.d_es),
            ("d_gsf", &rm.d_gsf),
            ("d_st", &rm.d_st),
            ("nu_st", &rm.nu_st),
            ("omega_shear", &rm.omega_shear),
            ("dynamo_b_r", &rm.dynamo_b_r),
            ("dynamo_b_phi", &rm.dynamo_b_phi),
        ] {
            assert_eq!(arr.len(), nz, "{name} len mismatch");
            assert!(arr.iter().all(|&v| v == 0.0), "{name} not zeroed");
        }

        // Second call clears pre-existing non-zero sentinels in the 13
        // `rotmix` arrays and 7 `am_*`/`d_mix_non_rotation` arrays, while
        // leaving `d_omega`, `grad_density`, `grad_temperature`, and
        // `d_mix_rotation` untouched (`hydro_rotation.f90:509-531`).
        model.aux.need_to_setvars = false;
        let aux = &mut model.aux;
        for arr in [
            &mut aux.am_nu_rot,
            &mut aux.am_nu_non_rot,
            &mut aux.am_nu_omega,
            &mut aux.am_nu_j,
            &mut aux.am_sig_omega,
            &mut aux.am_sig_j,
            &mut aux.mixing.d_mix_non_rotation,
            &mut aux.rotmix.domega_dlnr,
            &mut aux.rotmix.richardson_number,
            &mut aux.rotmix.d_visc,
            &mut aux.rotmix.d_dsi,
            &mut aux.rotmix.d_sh,
            &mut aux.rotmix.d_ssi,
            &mut aux.rotmix.d_es,
            &mut aux.rotmix.d_gsf,
            &mut aux.rotmix.d_st,
            &mut aux.rotmix.nu_st,
            &mut aux.rotmix.omega_shear,
            &mut aux.rotmix.dynamo_b_r,
            &mut aux.rotmix.dynamo_b_phi,
        ] {
            arr.fill(42.0);
        }
        aux.rotmix.d_omega = vec![1.25; nz];
        aux.rotmix.grad_density = vec![2.5; nz];
        aux.rotmix.grad_temperature = vec![3.75; nz];
        aux.rotmix.d_mix_rotation = vec![5.0; nz];

        apply_uniform_rotation(&mut model, 2.0e-5);
        assert!(model.aux.need_to_setvars);

        let aux = &model.aux;
        for (name, arr) in [
            ("am_nu_rot", &aux.am_nu_rot),
            ("am_nu_non_rot", &aux.am_nu_non_rot),
            ("am_nu_omega", &aux.am_nu_omega),
            ("am_nu_j", &aux.am_nu_j),
            ("am_sig_omega", &aux.am_sig_omega),
            ("am_sig_j", &aux.am_sig_j),
            ("d_mix_non_rotation", &aux.mixing.d_mix_non_rotation),
            ("domega_dlnr", &aux.rotmix.domega_dlnr),
            ("richardson_number", &aux.rotmix.richardson_number),
            ("d_visc", &aux.rotmix.d_visc),
            ("d_dsi", &aux.rotmix.d_dsi),
            ("d_sh", &aux.rotmix.d_sh),
            ("d_ssi", &aux.rotmix.d_ssi),
            ("d_es", &aux.rotmix.d_es),
            ("d_gsf", &aux.rotmix.d_gsf),
            ("d_st", &aux.rotmix.d_st),
            ("nu_st", &aux.rotmix.nu_st),
            ("omega_shear", &aux.rotmix.omega_shear),
            ("dynamo_b_r", &aux.rotmix.dynamo_b_r),
            ("dynamo_b_phi", &aux.rotmix.dynamo_b_phi),
        ] {
            assert!(
                arr.iter().all(|&v| v == 0.0),
                "{name} sentinel was not zeroed"
            );
        }

        assert!(aux
            .rotmix
            .d_omega
            .iter()
            .all(|&v| v.to_bits() == 1.25_f64.to_bits()));
        assert!(aux
            .rotmix
            .grad_density
            .iter()
            .all(|&v| v.to_bits() == 2.5_f64.to_bits()));
        assert!(aux
            .rotmix
            .grad_temperature
            .iter()
            .all(|&v| v.to_bits() == 3.75_f64.to_bits()));
        assert!(aux
            .rotmix
            .d_mix_rotation
            .iter()
            .all(|&v| v.to_bits() == 5.0_f64.to_bits()));
    }
}
