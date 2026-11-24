// SPDX-License-Identifier: GPL-3.0-or-later
//! Convective boundary detection, mixing glitch removal, envelope mixing,
//! minimum-diffusivity floors, crossover zeroing, face-centered Lagrangian
//! diffusion coupling factors ($\sigma_k$), and conservative species mixing
//! rates (`star/private/mix_info.f90`).

use flexmesa_const::{MixingType, FOUR_PI, LN10, M_SUN, SQRT_2_DIV_3};

use crate::controls::StarError;
use crate::model::StarModel;

const CAT_PP: usize = 0;
const CAT_CNO: usize = 1;
const CAT_3ALF: usize = 2;

/// Upstream `controls_dev.defaults:44-45` RSP2 mixing-type luminosity ratios.
const RSP2_MIN_LT_DIV_L_FOR_OVERSHOOTING_MIXING_TYPE: f64 = 1.0e-4;
const RSP2_MIN_LC_DIV_L_FOR_CONVECTIVE_MIXING_TYPE: f64 = 1.0e-4;

/// Computes the face-interpolated density $\rho_{\mathrm{face},k}$ at interior
/// face $k \in \{1, \dots, \mathrm{nz}-1\}$ (`mix_info.f90:118-119`).
#[must_use]
#[inline]
pub fn face_density(model: &StarModel, k: usize) -> f64 {
    if k >= model.nz {
        model.rho.first().copied().unwrap_or(0.0)
    } else if k == 0 {
        model.rho[0]
    } else {
        // `rho_face = (s% dq(k-1)*s% rho(k) + s% dq(k)*s% rho(k-1))/
        // (s% dq(k-1) + s% dq(k))` (`mix_info.f90:118-119`).
        (model.dq[k - 1] * model.rho[k] + model.dq[k] * model.rho[k - 1])
            / (model.dq[k - 1] + model.dq[k])
    }
}

/// Computes the geometric factor $(4\pi r_k^2 \rho_{\mathrm{face},k})^2$ at
/// face $k$ (`cdc_factor` in `mix_info.f90:116-122`).
#[must_use]
#[inline]
pub fn cdc_factor(model: &StarModel, k: usize) -> f64 {
    if k == 0 || k >= model.nz {
        1.0
    } else {
        let rho_f = face_density(model, k);
        let area_rho = FOUR_PI * model.r[k] * model.r[k] * rho_f;
        area_rho * area_rho
    }
}

/// Runs the complete MESA `26.4.1` `set_mixing_info` pipeline with
/// `skip_set_cz_bdy_mass = false` (`star/private/mix_info.f90:48-425`).
///
/// # Errors
/// Returns [`StarError`] if RTI/RSP2 arrays or `eps_nuc_categories` are
/// unsized, or if predictive mixing, overshooting, or boundary checks fail or
/// produce non-finite `D_mix`.
pub fn set_mixing_info(model: &mut StarModel) -> Result<(), StarError> {
    set_mixing_info_with_skip(model, false)
}

/// Runs the complete MESA `26.4.1` `set_mixing_info` pipeline
/// (`star/private/mix_info.f90:48-425`):
/// 1. When `RTI_flag` is set, calls `set_rti_mixing_info` and
///    `set_dpdr_drhodr_info` (`mix_info.f90:79-84`).
/// 2. Copies `mlt_mixing_type`, `mlt_D`, `mlt_cdc`, `mlt_vc` into
///    `mixing_type`, `d_mix`, `cdc`, `conv_vel` (or evaluates the `RSP_flag` /
///    `RSP2_flag` branches at `mix_info.f90:120-149`).
/// 3. Runs the 7 glitch-cleanup passes when `remove_mixing_glitches` and not
///    `RSP2_or_RSP`.
/// 4. Runs `do_mix_envelope` (`T_mix_limit`).
/// 5. Runs `set_cz_boundary_info`, `locate_convection_boundaries`,
///    `add_predictive_mixing`, re-runs `locate_convection_boundaries`, runs
///    `locate_mixing_boundaries`, runs `add_overshooting`, and calls
///    `add_rti_turbulence` (`mix_info.f90:230-231, 2373-2436`).
/// 6. Updates `set_cz_bdy_mass` unless `skip_set_cz_bdy_mass || RSP2_or_RSP`
///    (`mix_info.f90:236-239`, `hydro_vars.f90:81`).
/// 7. Applies `min_D_mix`, `min_D_mix_below_Tmax`, `min_D_mix_in_H_He`.
/// 8. Saves `d_mix_non_rotation` and adds rotational mixing if enabled.
/// 9. Applies `D_mix_zero_region_*`, H/He and H/C crossover zeroing.
/// 10. Updates `conv_vel` from `d_mix` and sets surface plotting values.
///
/// Upstream also calls `other_after_set_mixing_info` (`mix_info.f90:233-234`)
/// and `other_D_mix` when `use_other_D_mix` (`:275-278`); neither hook is
/// ported (`use_other_D_mix` is rejected as an unsupported control during
/// strict inlist validation, and no MESA `26.4.1` test case installs
/// `other_after_set_mixing_info`).
///
/// # Errors
/// Returns [`StarError`] if required RSP2, RTI, or nuclear-burn arrays are
/// unsized, or if predictive mixing, overshooting, RTI turbulence, or boundary
/// checks fail or produce non-finite `D_mix`.
pub fn set_mixing_info_with_skip(
    model: &mut StarModel,
    skip_set_cz_bdy_mass: bool,
) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.aux.mixing.cdc.len() != nz {
        model.aux.mixing.resize(nz);
    }

    if model.controls.rti_flag {
        crate::hydro_rti::set_rti_mixing_info(model);
        crate::hydro_rti::set_dpdr_drhodr_info(model);
    }

    let rsp2_or_rsp = model.controls.rsp_flag || model.controls.rsp2_flag;
    if !rsp2_or_rsp {
        model
            .mixing_type
            .copy_from_slice(&model.aux.mixing.mlt_mixing_type[..nz]);
    }

    let mut cdc_fac = vec![1.0_f64; nz];
    for k in 1..nz {
        let rho_face = (model.dq[k - 1] * model.rho[k]
            + model.dq[k] * model.rho[k - 1])
            / (model.dq[k - 1] + model.dq[k]);
        let f = FOUR_PI * model.r[k] * model.r[k] * rho_face;
        cdc_fac[k] = f * f;
    }

    if model.controls.rsp_flag {
        model.mixing_type[..nz].fill(MixingType::NoMixing);
        model.d_mix[..nz].fill(0.0);
        model.aux.mixing.cdc[..nz].fill(0.0);
        model.conv_vel[..nz].fill(0.0);
    } else if model.controls.rsp2_flag {
        // Upstream `star/private/mix_info.f90:126-142` and
        // `star/private/auto_diff_support.f90:402-413` (`get_RSP2_conv_velocity`).
        if model.aux.rsp.et_rsp.len() != nz
            || model.aux.rsp.hp_face.len() != nz
            || model.aux.rsp.lc_rsp.len() != nz
            || model.aux.rsp.lt_rsp.len() != nz
        {
            return Err(StarError::InvalidInput(format!(
                "set_mixing_info: RSP2 arrays unsized (nz={nz}, et={}, hp_face={}, lc={}, lt={})",
                model.aux.rsp.et_rsp.len(),
                model.aux.rsp.hp_face.len(),
                model.aux.rsp.lc_rsp.len(),
                model.aux.rsp.lt_rsp.len()
            )));
        }
        let alpha = model.controls.mixing_length_alpha;
        for k in 0..nz {
            let cv = if k == 0 {
                0.0
            } else {
                let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
                let beta = 1.0 - alfa;
                let w_k = model.aux.rsp.et_rsp[k].max(0.0).sqrt();
                let w_km1 = model.aux.rsp.et_rsp[k - 1].max(0.0).sqrt();
                SQRT_2_DIV_3 * (alfa * w_k + beta * w_km1)
            };
            model.conv_vel[k] = cv;
            model.d_mix[k] = cv * alpha * model.aux.rsp.hp_face[k] / 3.0;
            model.aux.mixing.cdc[k] = cdc_fac[k] * model.d_mix[k];
            let l_val = 1.0e-99_f64.max(model.lum[k].abs());
            if model.aux.rsp.lt_rsp[k].abs()
                > l_val * RSP2_MIN_LT_DIV_L_FOR_OVERSHOOTING_MIXING_TYPE
            {
                model.mixing_type[k] = MixingType::OvershootMixing;
            } else if model.aux.rsp.lc_rsp[k].abs()
                > l_val * RSP2_MIN_LC_DIV_L_FOR_CONVECTIVE_MIXING_TYPE
            {
                model.mixing_type[k] = MixingType::ConvectiveMixing;
            } else {
                model.mixing_type[k] = MixingType::NoMixing;
            }
        }
    } else {
        model.d_mix[..nz].copy_from_slice(&model.mlt_d[..nz]);
        model.aux.mixing.cdc[..nz].copy_from_slice(&model.mlt_cdc[..nz]);
        model.conv_vel[..nz].copy_from_slice(&model.mlt_vc[..nz]);
    }

    check_d_mix_finite(model, "after get mlt_D")?;

    if model.controls.remove_mixing_glitches && !rsp2_or_rsp {
        clean_mixing_glitches_from_controls(model);
    }

    check_d_mix_finite(model, "after remove_mixing_glitches")?;

    do_mix_envelope(model);

    let (eps_h, eps_he, eps_z) = extract_burn_eps(model)?;

    if !model.controls.rsp_flag {
        set_cz_boundary_info(model)?;
        locate_convection_boundaries_in_place(model, &eps_h, &eps_he, &eps_z);

        // Predictive mixing (`star/private/predictive_mix.f90`).
        let _ = crate::predictive_mix::add_predictive_mixing(model)?;
    }

    check_d_mix_finite(model, "after add_predictive_mixing")?;

    // Re-locate convection boundaries after `add_predictive_mixing`, then
    // locate all mixing boundaries and apply overshooting (`mix_info.f90:213-226`).
    if !model.controls.rsp_flag {
        locate_convection_boundaries_in_place(model, &eps_h, &eps_he, &eps_z);
        locate_mixing_boundaries_in_place(model);
        crate::overshoot::add_overshooting(model)?;
    }

    check_d_mix_finite(model, "after add_overshooting")?;

    add_rti_turbulence(model)?;

    if !(skip_set_cz_bdy_mass || rsp2_or_rsp) {
        set_cz_bdy_mass(model);
    }

    apply_min_d_mix_floor(model);
    apply_min_d_mix_below_tmax(model);
    apply_min_d_mix_in_h_he(model);

    model
        .aux
        .mixing
        .d_mix_non_rotation
        .copy_from_slice(&model.d_mix[..nz]);

    check_d_mix_finite(model, "before rotation_flag")?;

    // Verbatim `star/private/mix_info.f90:287-301`: gate on `rotation_flag`,
    // clip `D_mix < 1e-10` and refresh `cdc`, then mark `RotationMixing` iff
    // `D_mix /= 0 && D_mix_non_rotation < D_mix_rotation`.
    if model.controls.rotation_flag {
        let d_mix_rotation =
            crate::rotation_mix::update_rotation_mixing_info(model)?;
        for k in 1..nz {
            if model.d_mix[k] < 1.0e-10 {
                model.d_mix[k] = 0.0;
            }
            model.aux.mixing.cdc[k] = model.d_mix[k] * cdc_fac[k];
            if model.d_mix[k] != 0.0
                && model.aux.mixing.d_mix_non_rotation[k] < d_mix_rotation[k]
            {
                model.mixing_type[k] = MixingType::RotationMixing;
            }
        }
        if nz >= 2 {
            model.aux.mixing.cdc[0] = model.aux.mixing.cdc[1];
        }
    }

    check_d_mix_finite(model, "after update_rotation_mixing_info")?;

    apply_d_mix_zero_regions(model);

    // Final `conv_vel` update from `D_mix` and mixing length (`mix_info.f90:364-376`).
    let has_alpha = model.aux.mixing.alpha_mlt.len() == nz
        && model.aux.mixing.alpha_mlt.iter().any(|&v| v > 0.0);
    for k in 1..nz {
        let alpha_k = if has_alpha {
            model.aux.mixing.alpha_mlt[k]
        } else {
            model.controls.mixing_length_alpha
        };
        let lam = alpha_k * model.scale_height[k];
        if lam > 0.0 {
            model.conv_vel[k] = 3.0 * model.d_mix[k] / lam;
        } else {
            model.conv_vel[k] = 0.0;
        }
    }
    if nz >= 2 {
        model.mixing_type[0] = model.mixing_type[1];
        model.d_mix[0] = model.d_mix[1];
        model.conv_vel[0] = 0.0;
    }

    check_d_mix_finite(model, "final")?;
    model.aux.mixing.have_mixing_info = true;
    Ok(())
}

fn check_d_mix_finite(model: &StarModel, stage: &str) -> Result<(), StarError> {
    for k in 0..model.nz {
        if !model.d_mix[k].is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "set_mixing_info ({stage}): non-finite D_mix at face {k}: {}",
                model.d_mix[k]
            )));
        }
    }
    Ok(())
}

/// Removes single-zone mixing glitches, tiny diffusion coefficients, narrow
/// radiative gaps, thermohaline dropouts, and embedded semiconvective zones
/// (`remove_mixing_glitches` in `star/private/mix_info.f90:157-180,969-1341`).
pub fn clean_mixing_glitches(
    model: &mut StarModel,
    clip_d_limit: f64,
    min_convective_gap_hp: f64,
) {
    if model.nz < 3 {
        return;
    }
    sync_cdc_if_zero(model);
    remove_tiny_mixing(model, clip_d_limit);
    remove_mixing_singletons(model, true);
    if min_convective_gap_hp >= 0.0 {
        close_gaps(model, MixingType::ConvectiveMixing, min_convective_gap_hp);
        close_gaps(
            model,
            MixingType::ThermohalineMixing,
            min_convective_gap_hp,
        );
    }
    remove_embedded_semiconvection_zones(model);
}

/// Applies the 7 MESA `26.4.1` mixing glitch cleanup passes parameterized by
/// `model.controls.mix_wind` (`mix_info.f90:157-180,969-1341`).
pub fn clean_mixing_glitches_from_controls(model: &mut StarModel) {
    if model.nz < 3 {
        return;
    }
    sync_cdc_if_zero(model);
    let mw = model.controls.mix_wind.clone();
    remove_tiny_mixing(model, mw.clip_d_limit);
    remove_mixing_singletons(model, mw.okay_to_remove_mixing_singleton);
    close_gaps(model, MixingType::ConvectiveMixing, mw.min_convective_gap);
    close_gaps(
        model,
        MixingType::ThermohalineMixing,
        mw.min_thermohaline_gap,
    );
    if mw.min_thermohaline_dropout > 0.0 {
        remove_thermohaline_dropouts(
            model,
            mw.min_thermohaline_dropout,
            mw.max_dropout_gradl_sub_grada,
        );
    }
    close_gaps(
        model,
        MixingType::SemiconvectiveMixing,
        mw.min_semiconvection_gap,
    );
    if mw.remove_embedded_semiconvection {
        remove_embedded_semiconvection_zones(model);
    }
}

fn sync_cdc_if_zero(model: &mut StarModel) {
    let nz = model.nz;
    if model.aux.mixing.cdc.len() != nz {
        model.aux.mixing.resize(nz);
    }
    // Support standalone unit tests that populate `model.d_mix` directly
    // without going through `set_mixing_info`.
    let cdc_all_zero = model.aux.mixing.cdc.iter().all(|&c| c == 0.0);
    let d_any_nonzero = model.d_mix.iter().any(|&d| d > 0.0);
    if cdc_all_zero && d_any_nonzero {
        for k in 0..nz {
            model.aux.mixing.cdc[k] = cdc_factor(model, k) * model.d_mix[k];
        }
    }
}

fn remove_tiny_mixing(model: &mut StarModel, tiny: f64) {
    // Verbatim `star/private/mix_info.f90:984-992`: does not alter `gradT`
    // or `mlt_*`.
    if tiny <= 0.0 {
        return;
    }
    for k in 0..model.nz {
        if model.d_mix[k] < tiny {
            model.aux.mixing.cdc[k] = 0.0;
            model.d_mix[k] = 0.0;
            model.conv_vel[k] = 0.0;
            model.mixing_type[k] = MixingType::NoMixing;
        }
    }
}

fn remove_mixing_singletons(model: &mut StarModel, remove_conv: bool) {
    // Verbatim `star/private/mix_info.f90:999-1076`.
    let nz = model.nz;
    let has_alpha = model.aux.mixing.alpha_mlt.len() == nz
        && model.aux.mixing.alpha_mlt.iter().any(|&v| v > 0.0);
    for k in 1..(nz - 1) {
        if model.aux.mixing.cdc[k] == 0.0 {
            if model.aux.mixing.cdc[k - 1] != 0.0
                && model.aux.mixing.cdc[k + 1] != 0.0
            {
                let cdc_k =
                    (model.aux.mixing.cdc[k - 1] + model.aux.mixing.cdc[k + 1])
                        / 2.0;
                model.aux.mixing.cdc[k] = cdc_k;
                let denom = flexmesa_math::pow2(
                    FOUR_PI * model.r[k] * model.r[k] * model.rho[k],
                );
                model.d_mix[k] = cdc_k / denom;
                let alpha_k = if has_alpha {
                    model.aux.mixing.alpha_mlt[k]
                } else {
                    model.controls.mixing_length_alpha
                };
                let lambda = alpha_k
                    * (model.scale_height[k - 1] + model.scale_height[k + 1])
                    / 2.0;
                model.conv_vel[k] = if lambda > 0.0 {
                    3.0 * model.d_mix[k] / lambda
                } else {
                    0.0
                };
                let code = model.mixing_type[k - 1]
                    .code()
                    .max(model.mixing_type[k + 1].code());
                model.mixing_type[k] = MixingType::from_code(code)
                    .unwrap_or(MixingType::ConvectiveMixing);
            }
        } else if remove_conv
            && model.aux.mixing.cdc[k - 1] == 0.0
            && model.aux.mixing.cdc[k + 1] == 0.0
        {
            crate::hydro_vars::switch_to_radiative(model, k);
            model.aux.mixing.cdc[k] = 0.0;
            model.d_mix[k] = 0.0;
            model.conv_vel[k] = 0.0;
            model.mixing_type[k] = MixingType::NoMixing;
        }
    }

    // Surface singleton (`mix_info.f90:1038-1055`).
    if model.aux.mixing.cdc[0] == 0.0 {
        if model.aux.mixing.cdc[1] != 0.0 {
            model.aux.mixing.cdc[0] = model.aux.mixing.cdc[1];
            model.d_mix[0] = model.d_mix[1];
            model.conv_vel[0] = model.conv_vel[1];
            model.mixing_type[0] = model.mixing_type[1];
        }
    } else if model.aux.mixing.cdc[1] == 0.0 {
        crate::hydro_vars::switch_to_radiative(model, 0);
        model.aux.mixing.cdc[0] = 0.0;
        model.d_mix[0] = 0.0;
        model.conv_vel[0] = 0.0;
        model.mixing_type[0] = MixingType::NoMixing;
    }

    // Center singleton (`mix_info.f90:1057-1074`).
    let last = nz - 1;
    if model.aux.mixing.cdc[last] == 0.0 {
        if model.aux.mixing.cdc[last - 1] != 0.0 {
            model.aux.mixing.cdc[last] = model.aux.mixing.cdc[last - 1];
            model.d_mix[last] = model.d_mix[last - 1];
            model.conv_vel[last] = model.conv_vel[last - 1];
            model.mixing_type[last] = model.mixing_type[last - 1];
        }
    } else if model.aux.mixing.cdc[last - 1] == 0.0 {
        crate::hydro_vars::switch_to_radiative(model, last);
        model.aux.mixing.cdc[last] = 0.0;
        model.d_mix[last] = 0.0;
        model.conv_vel[last] = 0.0;
        model.mixing_type[last] = MixingType::NoMixing;
    }
}

fn close_gaps(model: &mut StarModel, mix_type: MixingType, min_gap: f64) {
    // Verbatim `star/private/mix_info.f90:1112-1176`.
    if min_gap < 0.0 || model.nz < 3 {
        return;
    }
    let nz = model.nz;
    let mut in_region = model.mixing_type[nz - 1] == mix_type;
    let mut rbot = 0.0_f64;
    let mut kbot = nz - 1;
    for k in (1..(nz - 1)).rev() {
        if in_region {
            if model.mixing_type[k] != mix_type {
                kbot = k + 1;
                rbot = model.r[kbot];
                in_region = false;
            }
        } else if model.mixing_type[k] == mix_type {
            let ktop = k;
            let rtop = model.r[ktop];
            let hp =
                model.peos[ktop] / (model.rho[ktop] * model.grav[ktop]);
            if rtop - rbot < hp * min_gap {
                if kbot < nz - 1 {
                    let cdc_avg = (model.aux.mixing.cdc[ktop]
                        + model.aux.mixing.cdc[kbot])
                        / 2.0;
                    let d_avg = (model.d_mix[ktop] + model.d_mix[kbot]) / 2.0;
                    let vc_avg =
                        (model.conv_vel[ktop] + model.conv_vel[kbot]) / 2.0;
                    for j in (ktop + 1)..kbot {
                        model.aux.mixing.cdc[j] = cdc_avg;
                        model.d_mix[j] = d_avg;
                        model.conv_vel[j] = vc_avg;
                        model.mixing_type[j] = mix_type;
                    }
                } else {
                    let cdc_t = model.aux.mixing.cdc[ktop];
                    let d_t = model.d_mix[ktop];
                    let vc_t = model.conv_vel[ktop];
                    for j in (ktop + 1)..=kbot {
                        model.aux.mixing.cdc[j] = cdc_t;
                        model.d_mix[j] = d_t;
                        model.conv_vel[j] = vc_t;
                        model.mixing_type[j] = mix_type;
                    }
                }
            }
            in_region = true;
        }
    }
}

fn remove_thermohaline_dropouts(
    model: &mut StarModel,
    min_dropout_hp: f64,
    max_gradl_sub_grada: f64,
) {
    // Verbatim `star/private/mix_info.f90:1183-1237`.
    let nz = model.nz;
    if nz < 4 {
        return;
    }
    // `s% grada_face(k)` (`mix_info.f90:1203, 1232-1233`).
    let grada_at = |m: &StarModel, idx: usize| -> f64 {
        m.aux.mixing.grada_face[idx]
    };

    let mut rbot = model.r[nz - 1];
    let mut kbot = nz - 2;
    let mut in_region = model.mixing_type[kbot] == MixingType::NoMixing;
    let mut all_small = false;

    for k in (1..(nz - 2)).rev() {
        if in_region {
            if model.mixing_type[k] == MixingType::NoMixing {
                if model.gradl[k] - grada_at(model, k) > max_gradl_sub_grada {
                    all_small = false;
                }
            } else {
                let ktop = k + 1;
                let rtop = model.r[ktop];
                let hp = model.peos[ktop]
                    / (model.rho[ktop] * model.grav[ktop]);
                let q_upper = model.q[ktop - 1];
                let q_lower = model.q[kbot + 1];
                if rtop - rbot < hp * min_dropout_hp
                    && model.mixing_type[ktop - 1]
                        == MixingType::ThermohalineMixing
                    && model.mixing_type[kbot + 1]
                        == MixingType::ThermohalineMixing
                    && q_upper - q_lower > 1.0e-20
                    && all_small
                {
                    for j in ktop..=kbot {
                        let alfa = (model.q[j] - q_lower) / (q_upper - q_lower);
                        let beta = 1.0 - alfa;
                        model.aux.mixing.cdc[j] = alfa
                            * model.aux.mixing.cdc[ktop - 1]
                            + beta * model.aux.mixing.cdc[kbot + 1];
                        model.d_mix[j] = alfa * model.d_mix[ktop - 1]
                            + beta * model.d_mix[kbot + 1];
                        model.conv_vel[j] = alfa * model.conv_vel[ktop - 1]
                            + beta * model.conv_vel[kbot + 1];
                        model.mixing_type[j] = MixingType::ThermohalineMixing;
                    }
                }
                in_region = false;
            }
        } else if model.mixing_type[k] == MixingType::NoMixing {
            kbot = k;
            rbot = model.r[kbot];
            in_region = true;
            all_small =
                model.gradl[k] - grada_at(model, k) <= max_gradl_sub_grada;
        }
    }
}

fn remove_embedded_semiconvection_zones(model: &mut StarModel) {
    // Verbatim `star/private/mix_info.f90:1240-1341`.
    let nz = model.nz;
    if nz < 3 {
        return;
    }
    let check = |mt: MixingType| {
        mt == MixingType::SemiconvectiveMixing
            || mt == MixingType::ConvectiveMixing
    };

    let mut in_region = check(model.mixing_type[nz - 1]);
    let mut kbot = nz - 1;
    for k in (1..(nz - 1)).rev() {
        if in_region {
            if !check(model.mixing_type[k]) {
                let ktop = k + 1;
                in_region = false;
                clean_embedded_region(model, ktop, kbot);
            }
        } else if check(model.mixing_type[k]) {
            kbot = k;
            in_region = true;
        }
    }
    if in_region {
        clean_embedded_region(model, 0, kbot);
    }
}

fn clean_embedded_region(model: &mut StarModel, mut ktop: usize, mut kbot: usize) {
    while model.mixing_type[ktop] == MixingType::ConvectiveMixing {
        ktop += 1;
        if ktop >= kbot {
            return;
        }
    }
    while model.mixing_type[ktop] == MixingType::SemiconvectiveMixing {
        ktop += 1;
        if ktop >= kbot {
            return;
        }
    }
    while model.mixing_type[kbot] == MixingType::ConvectiveMixing {
        if kbot == 0 {
            return;
        }
        kbot -= 1;
        if ktop >= kbot {
            return;
        }
    }
    while model.mixing_type[kbot] == MixingType::SemiconvectiveMixing {
        if kbot == 0 {
            return;
        }
        kbot -= 1;
        if ktop >= kbot {
            return;
        }
    }

    let mut kbot1 = kbot;
    while kbot1 > ktop {
        while model.mixing_type[kbot1] == MixingType::ConvectiveMixing {
            if kbot1 <= ktop {
                return;
            }
            kbot1 -= 1;
            if kbot1 <= ktop {
                return;
            }
        }
        let mut ktop1 = kbot1;
        while model.mixing_type[ktop1] == MixingType::SemiconvectiveMixing {
            if ktop1 <= ktop {
                return;
            }
            ktop1 -= 1;
            if ktop1 <= ktop {
                return;
            }
        }
        let d_top = model.d_mix[ktop1];
        let vc_top = model.conv_vel[ktop1];
        for j in (ktop1 + 1)..=kbot1 {
            model.d_mix[j] = d_top;
            model.mixing_type[j] = MixingType::ConvectiveMixing;
            model.conv_vel[j] = vc_top;
        }
        kbot1 = ktop1;
    }
}

/// Extends outer convective envelope mixing to the surface when
/// `T_mix_limit > 0` (`do_mix_envelope` in `star/private/mix_info.f90:1344-1381`).
pub fn do_mix_envelope(model: &mut StarModel) {
    let t_mix_limit = model.controls.mix_wind.t_mix_limit;
    if t_mix_limit <= 0.0 || model.nz == 0 {
        return;
    }
    let nz = model.nz;
    let mut j_opt: Option<usize> = None;
    for k in 0..nz {
        if model.t[k] >= t_mix_limit {
            j_opt = Some(k);
            break;
        }
    }
    let j = j_opt.unwrap_or(nz - 1);
    let mut i_opt: Option<usize> = None;
    for k in (0..=j).rev() {
        if model.mixing_type[k] == MixingType::ConvectiveMixing {
            i_opt = Some(k);
            break;
        }
    }
    let Some(i) = i_opt else {
        return;
    };
    let mut j_max = 0_usize;
    let mut max_cdc = f64::NEG_INFINITY;
    for k in 0..=i {
        if model.aux.mixing.cdc[k] > max_cdc {
            max_cdc = model.aux.mixing.cdc[k];
            j_max = k;
        }
    }
    let vc_ref = model.conv_vel[j_max];
    let cdc_ref = model.aux.mixing.cdc[j_max];
    let d_ref = model.d_mix[j_max];
    for k in 0..=i {
        model.conv_vel[k] = vc_ref;
        model.aux.mixing.cdc[k] = cdc_ref;
        model.d_mix[k] = d_ref;
        model.mixing_type[k] = MixingType::ConvectiveMixing;
    }
}

/// Computes sub-cell convective/semiconvective boundary mass offsets
/// `model.aux.mixing.cz_bdy_dq` (`set_cz_boundary_info` in
/// `star/private/mix_info.f90:428-469`).
///
/// Leaves `cz_bdy_dq[k - 1] = 0.0` when `dg0 * dg1 >= 0.0` (no half-cell
/// fallback).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `find0` produces an out-of-range
/// offset.
pub fn set_cz_boundary_info(model: &mut StarModel) -> Result<(), StarError> {
    let nz = model.nz;
    if model.aux.mixing.cz_bdy_dq.len() != nz {
        model.aux.mixing.resize(nz);
    }
    model.aux.mixing.cz_bdy_dq.fill(0.0);
    if nz < 2 {
        return Ok(());
    }
    for k in 1..nz {
        let mt1 = model.mixing_type[k - 1];
        let mt2 = model.mixing_type[k];
        if mt1 == mt2 {
            continue;
        }
        let (dg0, dg1) = if mt2 == MixingType::ConvectiveMixing
            || mt1 == MixingType::ConvectiveMixing
        {
            (
                model.gradl[k - 1] - model.gradr[k - 1],
                model.gradl[k] - model.gradr[k],
            )
        } else if mt2 == MixingType::SemiconvectiveMixing
            || mt1 == MixingType::SemiconvectiveMixing
        {
            // `dg0 = s% grada_face(k-1) - s% gradr(k-1)`,
            // `dg1 = s% grada_face(k) - s% gradr(k)` (`mix_info.f90:454-455`).
            let gf = &model.aux.mixing.grada_face;
            (gf[k - 1] - model.gradr[k - 1], gf[k] - model.gradr[k])
        } else {
            continue;
        };
        if dg0 * dg1 >= 0.0 {
            continue;
        }
        let x0 = flexmesa_num::find0(0.0, dg0, model.dq[k - 1], dg1).map_err(
            |e| {
                StarError::InvalidInput(format!(
                    "set_cz_boundary_info find0 failed at cell {}: {e}",
                    k - 1
                ))
            },
        )?;
        if x0 < 0.0 || x0 > model.dq[k - 1] {
            return Err(StarError::InvalidInput(format!(
                "bad cz_bdy_dq at cell {}: cz_bdy_dq={x0}, dq={}",
                k - 1,
                model.dq[k - 1]
            )));
        }
        model.aux.mixing.cz_bdy_dq[k - 1] = x0;
    }
    Ok(())
}

/// Populates `model.aux.mixing.{cz_bot_mass, cz_top_mass}` ordered center to
/// surface (`set_cz_bdy_mass` in `star/private/mix_info.f90:472-529`).
pub fn set_cz_bdy_mass(model: &mut StarModel) {
    let nz = model.nz;
    model.aux.mixing.cz_bot_mass.clear();
    model.aux.mixing.cz_top_mass.clear();
    if nz == 0 {
        model.aux.mixing.have_new_cz_bdy_info = true;
        return;
    }
    let mut in_conv = model.mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    if in_conv {
        model.aux.mixing.cz_bot_mass.push(model.m_center);
        model.aux.mixing.cz_top_mass.push(model.mstar);
    }
    if nz >= 3 {
        for k in (1..(nz - 1)).rev() {
            if in_conv {
                if model.mixing_type[k] != MixingType::ConvectiveMixing {
                    if let Some(last_top) =
                        model.aux.mixing.cz_top_mass.last_mut()
                    {
                        *last_top = model.m_center
                            + (model.q[k] - model.aux.mixing.cz_bdy_dq[k])
                                * model.xmstar;
                    }
                    in_conv = false;
                }
            } else if model.mixing_type[k] == MixingType::ConvectiveMixing {
                let bot_m = model.m_center
                    + (model.q[k] - model.aux.mixing.cz_bdy_dq[k])
                        * model.xmstar;
                model.aux.mixing.cz_bot_mass.push(bot_m);
                model.aux.mixing.cz_top_mass.push(model.mstar);
                in_conv = true;
            }
        }
    }
    if in_conv {
        if let Some(last_top) = model.aux.mixing.cz_top_mass.last_mut() {
            *last_top = model.mstar;
        }
    }
    model.aux.mixing.have_new_cz_bdy_info = true;
}

/// Locates convective boundaries in place on `model.aux.mixing`, prunes bad
/// thin convective zones (`prune_bad_cz_min_Hp_height`), and classifies H/He/Z
/// burning on each boundary pair (`locate_convection_boundaries` in
/// `star/private/mix_info.f90:532-788`).
pub(crate) fn locate_convection_boundaries_in_place(
    model: &mut StarModel,
    eps_h: &[f64],
    eps_he: &[f64],
    eps_z: &[f64],
) {
    let nz = model.nz;
    model.aux.mixing.conv_bdy_loc.clear();
    model.aux.mixing.conv_bdy_q.clear();
    model.aux.mixing.top_conv_bdy.clear();
    model.aux.mixing.burn_h_conv_region.clear();
    model.aux.mixing.burn_he_conv_region.clear();
    model.aux.mixing.burn_z_conv_region.clear();
    if nz < 2 {
        return;
    }

    let mut in_conv = model.mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    let mut k_bot = nz - 1;
    let mut bot_r = 0.0_f64;
    let mut bot_hp = 0.0_f64;

    if nz >= 3 {
        for k in (1..(nz - 1)).rev() {
            if in_conv {
                if model.mixing_type[k] != MixingType::ConvectiveMixing {
                    end_of_convective_region(
                        model, k, k_bot, bot_r, bot_hp, eps_h, eps_he, eps_z,
                    );
                    in_conv = false;
                }
            } else if model.mixing_type[k] == MixingType::ConvectiveMixing {
                k_bot = k + 1;
                let bdy_q = model.q[k] - model.aux.mixing.cz_bdy_dq[k];
                model.aux.mixing.conv_bdy_q.push(bdy_q);
                model.aux.mixing.top_conv_bdy.push(false);
                model.aux.mixing.conv_bdy_loc.push(k_bot);
                model.aux.mixing.burn_h_conv_region.push(false);
                model.aux.mixing.burn_he_conv_region.push(false);
                model.aux.mixing.burn_z_conv_region.push(false);
                in_conv = true;
                bot_r = model.r[k_bot];
                bot_hp = model.scale_height[k_bot];
            }
        }
    }
    if in_conv {
        end_of_convective_region(
            model, 0, k_bot, bot_r, bot_hp, eps_h, eps_he, eps_z,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn end_of_convective_region(
    model: &mut StarModel,
    k: usize,
    k_bot: usize,
    bot_r: f64,
    bot_hp: f64,
    eps_h: &[f64],
    eps_he: &[f64],
    eps_z: &[f64],
) {
    let nz = model.nz;
    let top_r = model.r[k];
    let top_hp = model.scale_height[k];
    let dr = top_r - bot_r;
    let hp = (bot_hp + top_hp) / 2.0;

    let min_hp = model.controls.mix_wind.prune_bad_cz_min_hp_height;
    if min_hp > 0.0 && hp > 0.0 && dr / hp < min_hp {
        let mut max_eps = f64::NEG_INFINITY;
        for kk in k..=k_bot {
            max_eps = max_eps.max(eps_h[kk] + eps_he[kk] + eps_z[kk]);
        }
        let min_eps =
            flexmesa_math::exp10(model.controls.mix_wind.prune_bad_cz_min_log_eps_nuc);
        let no_th = ((k + 1)..k_bot)
            .all(|kk| model.mixing_type[kk] != MixingType::ThermohalineMixing);
        if max_eps < min_eps && no_th {
            for kk in k..=k_bot {
                crate::hydro_vars::switch_to_radiative(model, kk);
                model.aux.mixing.cdc[kk] = 0.0;
                model.d_mix[kk] = 0.0;
                model.conv_vel[kk] = 0.0;
                model.mixing_type[kk] = MixingType::NoMixing;
            }
            if !model.aux.mixing.conv_bdy_loc.is_empty() {
                model.aux.mixing.conv_bdy_loc.pop();
                model.aux.mixing.conv_bdy_q.pop();
                model.aux.mixing.top_conv_bdy.pop();
                model.aux.mixing.burn_h_conv_region.pop();
                model.aux.mixing.burn_he_conv_region.pop();
                model.aux.mixing.burn_z_conv_region.pop();
            }
            return;
        }
    }

    let bdy_q = if k == 0 {
        1.0
    } else {
        model.q[k] - model.aux.mixing.cz_bdy_dq[k]
    };
    model.aux.mixing.conv_bdy_q.push(bdy_q);
    model.aux.mixing.top_conv_bdy.push(true);
    model.aux.mixing.conv_bdy_loc.push(k);
    model.aux.mixing.burn_h_conv_region.push(false);
    model.aux.mixing.burn_he_conv_region.push(false);
    model.aux.mixing.burn_z_conv_region.push(false);
    let i = model.aux.mixing.conv_bdy_loc.len() - 1;

    // Classify burning over `k..=min(nz-1, k_bot+1)` (`mix_info.f90:703-736`).
    let mut max_log_t = -1.0e99_f64;
    let mut max_x = 0.0_f64;
    let mut max_y = 0.0_f64;
    let k_hi = (k_bot + 1).min(nz - 1);
    for kk in k..=k_hi {
        let lt = model.ln_t[kk] / LN10;
        if lt > max_log_t {
            max_log_t = lt;
        }
        if model.xh_mass[kk] > max_x {
            max_x = model.xh_mass[kk];
        }
        if model.yhe_mass[kk] > max_y {
            max_y = model.yhe_mass[kk];
        }
    }

    let mw = &model.controls.mix_wind;
    if max_log_t > mw.burn_z_mix_region_logt
        && max_y < mw.max_y_for_burn_z_mix_region
    {
        model.aux.mixing.burn_z_conv_region[i] = true;
        if i > 0 {
            model.aux.mixing.burn_z_conv_region[i - 1] = true;
        }
    } else if max_log_t > mw.burn_he_mix_region_logt
        && max_x < mw.max_x_for_burn_he_mix_region
    {
        model.aux.mixing.burn_he_conv_region[i] = true;
        if i > 0 {
            model.aux.mixing.burn_he_conv_region[i - 1] = true;
        }
    } else if max_log_t > mw.burn_h_mix_region_logt {
        model.aux.mixing.burn_h_conv_region[i] = true;
        if i > 0 {
            model.aux.mixing.burn_h_conv_region[i - 1] = true;
        }
    }
}

/// Locates all non-zero mixing region boundaries (`locate_mixing_boundaries` in
/// `star/private/mix_info.f90:799-966`).
pub(crate) fn locate_mixing_boundaries_in_place(model: &mut StarModel) {
    let nz = model.nz;
    model.aux.mixing.mix_bdy_loc.clear();
    model.aux.mixing.mix_bdy_q.clear();
    model.aux.mixing.top_mix_bdy.clear();
    model.aux.mixing.burn_h_mix_region.clear();
    model.aux.mixing.burn_he_mix_region.clear();
    model.aux.mixing.burn_z_mix_region.clear();
    if nz < 2 {
        return;
    }

    let mut in_mix = model.mixing_type[nz - 1] != MixingType::NoMixing;
    let mut k_bot = nz - 1;

    if nz >= 3 {
        for k in (1..(nz - 1)).rev() {
            if in_mix {
                if model.mixing_type[k] == MixingType::NoMixing {
                    end_of_mixing_region(model, k, k_bot);
                    in_mix = false;
                }
            } else if model.mixing_type[k] != MixingType::NoMixing {
                k_bot = k + 1;
                let bdy_q = model.q[k] - model.aux.mixing.cz_bdy_dq[k];
                model.aux.mixing.mix_bdy_q.push(bdy_q);
                model.aux.mixing.top_mix_bdy.push(false);
                model.aux.mixing.mix_bdy_loc.push(k_bot);
                model.aux.mixing.burn_h_mix_region.push(false);
                model.aux.mixing.burn_he_mix_region.push(false);
                model.aux.mixing.burn_z_mix_region.push(false);
                in_mix = true;
            }
        }
    }
    if in_mix {
        end_of_mixing_region(model, 0, k_bot);
    }
}

fn end_of_mixing_region(model: &mut StarModel, k: usize, k_bot: usize) {
    let nz = model.nz;
    let bdy_q = if k == 0 {
        1.0
    } else {
        model.q[k] - model.aux.mixing.cz_bdy_dq[k]
    };
    model.aux.mixing.mix_bdy_q.push(bdy_q);
    model.aux.mixing.top_mix_bdy.push(true);
    model.aux.mixing.mix_bdy_loc.push(k);
    model.aux.mixing.burn_h_mix_region.push(false);
    model.aux.mixing.burn_he_mix_region.push(false);
    model.aux.mixing.burn_z_mix_region.push(false);
    let i = model.aux.mixing.mix_bdy_loc.len() - 1;

    let mut max_log_t = -1.0e99_f64;
    let mut max_x = 0.0_f64;
    let mut max_y = 0.0_f64;
    let k_hi = (k_bot + 1).min(nz - 1);
    for kk in k..=k_hi {
        let lt = model.ln_t[kk] / LN10;
        if lt > max_log_t {
            max_log_t = lt;
        }
        if model.xh_mass[kk] > max_x {
            max_x = model.xh_mass[kk];
        }
        if model.yhe_mass[kk] > max_y {
            max_y = model.yhe_mass[kk];
        }
    }
    let mw = &model.controls.mix_wind;
    if max_log_t > mw.burn_z_mix_region_logt
        && max_y < mw.max_y_for_burn_z_mix_region
    {
        model.aux.mixing.burn_z_mix_region[i] = true;
        if i > 0 {
            model.aux.mixing.burn_z_mix_region[i - 1] = true;
        }
    } else if max_log_t > mw.burn_he_mix_region_logt
        && max_x < mw.max_x_for_burn_he_mix_region
    {
        model.aux.mixing.burn_he_mix_region[i] = true;
        if i > 0 {
            model.aux.mixing.burn_he_mix_region[i - 1] = true;
        }
    } else if max_log_t > mw.burn_h_mix_region_logt {
        model.aux.mixing.burn_h_mix_region[i] = true;
        if i > 0 {
            model.aux.mixing.burn_h_mix_region[i - 1] = true;
        }
    }
}

/// Enforces `min_D_mix` across qualifying mass zones when `set_min_D_mix` is
/// enabled and central $Y_e \ge \mathrm{min\_center\_Ye\_for\_min\_D\_mix}$
/// (`mix_info.f90:241-249`).
pub fn apply_min_d_mix_floor(model: &mut StarModel) {
    let mw = &model.controls.mix_wind;
    let d_min = model.controls.min_d_mix;
    if !mw.set_min_d_mix || model.nz == 0 {
        return;
    }
    let center_ye = model.ye.last().copied().unwrap_or(0.0);
    if center_ye < mw.min_center_ye_for_min_d_mix {
        return;
    }
    let m_lo = mw.mass_lower_limit_for_min_d_mix * M_SUN;
    let m_hi = mw.mass_upper_limit_for_min_d_mix * M_SUN;
    for k in 0..model.nz {
        if model.d_mix[k] >= d_min || model.m[k] > m_hi || model.m[k] < m_lo {
            continue;
        }
        model.d_mix[k] = d_min;
        model.mixing_type[k] = MixingType::MinimumMixing;
    }
}

fn apply_min_d_mix_below_tmax(model: &mut StarModel) {
    // Verbatim `star/private/mix_info.f90:251-265`.
    let mw = &model.controls.mix_wind;
    if !mw.set_min_d_mix_below_tmax || model.nz == 0 {
        return;
    }
    let mut t_max = -1.0_f64;
    let mut k_tmax: Option<usize> = None;
    for k in 0..model.nz {
        if model.t[k] > t_max {
            t_max = model.t[k];
            k_tmax = Some(k);
        }
    }
    let Some(kt) = k_tmax else {
        return;
    };
    let d_min = mw.min_d_mix_below_tmax;
    for k in (kt + 1)..model.nz {
        if model.d_mix[k] >= d_min {
            continue;
        }
        model.d_mix[k] = d_min;
        model.mixing_type[k] = MixingType::MinimumMixing;
    }
}

fn apply_min_d_mix_in_h_he(model: &mut StarModel) {
    // Verbatim `star/private/mix_info.f90:267-274`.
    let mw = &model.controls.mix_wind;
    if !mw.set_min_d_mix_in_h_he || model.nz == 0 {
        return;
    }
    let d_min = mw.min_d_mix_in_h_he;
    for k in 0..model.nz {
        if model.xh_mass[k] + model.yhe_mass[k] < 0.5 {
            break;
        }
        if model.d_mix[k] >= d_min {
            continue;
        }
        model.d_mix[k] = d_min;
        model.mixing_type[k] = MixingType::MinimumMixing;
    }
}

fn apply_d_mix_zero_regions(model: &mut StarModel) {
    // Verbatim `star/private/mix_info.f90:305-360`.
    let nz = model.nz;
    if nz < 2 {
        return;
    }
    let mw = &model.controls.mix_wind;
    let mut region_bottom_q = mw.d_mix_zero_region_bottom_q;
    let mut region_top_q = mw.d_mix_zero_region_top_q;

    if mw.dq_d_mix_zero_at_h_he_crossover > 0.0 {
        let i_h1 = crate::report_utils::resolve_iso_slot(model, "h1")
            .ok()
            .and_then(|(_, s)| s);
        let i_he4 = crate::report_utils::resolve_iso_slot(model, "he4")
            .ok()
            .and_then(|(_, s)| s);
        if let (Some(jh), Some(jhe)) = (i_h1, i_he4) {
            let sp = model.species;
            for k in 1..nz {
                if model.xa[(k - 1) * sp + jh] > model.xa[(k - 1) * sp + jhe]
                    && model.xa[k * sp + jh] <= model.xa[k * sp + jhe]
                {
                    region_bottom_q =
                        model.q[k] - 0.5 * mw.dq_d_mix_zero_at_h_he_crossover;
                    region_top_q =
                        model.q[k] + 0.5 * mw.dq_d_mix_zero_at_h_he_crossover;
                    break;
                }
            }
        }
    }

    if region_bottom_q < region_top_q {
        for k in 1..nz {
            if model.q[k] >= region_bottom_q && model.q[k] <= region_top_q {
                model.d_mix[k] = 0.0;
                model.mixing_type[k] = MixingType::NoMixing;
            }
        }
    }

    let mut c_bottom_q = 1.0e99_f64;
    let mut c_top_q = -1.0e99_f64;
    if mw.dq_d_mix_zero_at_h_c_crossover > 0.0 {
        let i_h1 = crate::report_utils::resolve_iso_slot(model, "h1")
            .ok()
            .and_then(|(_, s)| s);
        let i_c12 = crate::report_utils::resolve_iso_slot(model, "c12")
            .ok()
            .and_then(|(_, s)| s);
        if let (Some(jh), Some(jc)) = (i_h1, i_c12) {
            let sp = model.species;
            for k in 1..nz {
                if model.xa[(k - 1) * sp + jh] > model.xa[(k - 1) * sp + jc]
                    && model.xa[k * sp + jh] <= model.xa[k * sp + jc]
                {
                    c_bottom_q =
                        model.q[k] - 0.5 * mw.dq_d_mix_zero_at_h_c_crossover;
                    c_top_q =
                        model.q[k] + 0.5 * mw.dq_d_mix_zero_at_h_c_crossover;
                    break;
                }
            }
        }
    }

    if c_bottom_q < c_top_q {
        for k in 1..nz {
            if model.q[k] >= c_bottom_q && model.q[k] <= c_top_q {
                model.d_mix[k] = 0.0;
                model.mixing_type[k] = MixingType::NoMixing;
            }
        }
    }
}

/// Extracts per-cell H, He, and metal burning rates from
/// `model.aux.eps_nuc_categories` (`star/private/mix_info.f90:186-191`).
fn extract_burn_eps(
    model: &StarModel,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), StarError> {
    let nz = model.nz;
    if model.aux.eps_nuc_categories.len() != nz {
        return Err(StarError::InvalidInput(format!(
            "extract_burn_eps: eps_nuc_categories has {} cells, expected nz = {nz}",
            model.aux.eps_nuc_categories.len()
        )));
    }
    let mut eps_h = vec![0.0_f64; nz];
    let mut eps_he = vec![0.0_f64; nz];
    let mut eps_z = vec![0.0_f64; nz];
    for k in 0..nz {
        let cats = &model.aux.eps_nuc_categories[k];
        eps_h[k] = cats[CAT_PP] + cats[CAT_CNO];
        eps_he[k] = cats[CAT_3ALF];
        eps_z[k] = model.eps_nuc[k] - (eps_h[k] + eps_he[k]);
    }
    Ok((eps_h, eps_he, eps_z))
}

/// Adds Rayleigh-Taylor turbulent diffusivity (`eta_RTI`) into `d_mix`, `cdc`,
/// `conv_vel`, and `mixing_type` (`add_RTI_turbulence` in
/// `star/private/mix_info.f90:2373-2436`).
///
/// Uses the upstream `controls.defaults:3127-3128, 8596-8597` defaults for the
/// mass taper and floor (`max_M_RTI_factors_full_on = 1e99`,
/// `min_M_RTI_factors_full_off = 1e99`, `RTI_D_mix_floor = 0`,
/// `RTI_min_m_for_D_mix_floor = 0`); inlists setting non-default values for
/// those controls fail loudly during strict validation.
fn add_rti_turbulence(model: &mut StarModel) -> Result<(), StarError> {
    if !model.controls.rti_flag {
        return Ok(());
    }
    let coeff = model.controls.composition_rti_diffusion_factor;
    if coeff <= 0.0 {
        return Ok(());
    }
    let nz = model.nz;
    if model.aux.eta_rti.len() != nz {
        return Err(StarError::InvalidInput(format!(
            "add_rti_turbulence: eta_rti has length {}, expected nz = {nz}",
            model.aux.eta_rti.len()
        )));
    }
    let has_alpha = model.aux.mixing.alpha_mlt.len() == nz
        && model.aux.mixing.alpha_mlt.iter().any(|&v| v > 0.0);
    for k in 1..nz {
        let mut d = model.aux.eta_rti[k] * coeff;
        if !d.is_finite() {
            return Err(StarError::MicrophysicsError(format!(
                "add_rti_turbulence: non-finite eta_RTI at face {k}: {d}"
            )));
        }
        if d > model.d_mix[k] {
            model.mixing_type[k] = MixingType::RayleighTaylorMixing;
        }
        d += model.d_mix[k];
        let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
        let beta = 1.0 - alfa;
        let rho_face = alfa * model.rho[k] + beta * model.rho[k - 1];
        let p_face = alfa * model.peos[k] + beta * model.peos[k - 1];
        let r_face = model.r[k];
        let q_face = model.q[k];
        let f = FOUR_PI * r_face * r_face * rho_face;
        let cdc = f * f * d;
        let hp = if model.cgrav[k] <= 0.0 {
            r_face
        } else {
            p_face
                / (rho_face
                    * model.cgrav[k]
                    * (model.m_center + model.xmstar * q_face)
                    / (r_face * r_face))
        };
        let alpha_k = if has_alpha {
            model.aux.mixing.alpha_mlt[k]
        } else {
            model.controls.mixing_length_alpha
        };
        let vc = 3.0 * d / (alpha_k * hp);
        model.aux.mixing.cdc[k] = cdc;
        model.d_mix[k] = d;
        model.conv_vel[k] = vc;
    }
    Ok(())
}

/// Populates sub-cell boundary fractions (`set_cz_boundary_info`) and locates
/// all convective boundaries in place on `model.aux.mixing`
/// (`set_cz_boundary_info` and `locate_convection_boundaries` in
/// `star/private/mix_info.f90:428-469, 532-788`).
///
/// Returns the number of convective boundaries stored in
/// `model.aux.mixing.conv_bdy_loc`.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if `eps_nuc_categories` is unsized or
/// if `set_cz_boundary_info` fails `find0` or produces out-of-range
/// `cz_bdy_dq`.
pub fn locate_convection_boundaries(
    model: &mut StarModel,
) -> Result<usize, StarError> {
    let nz = model.nz;
    if nz == 0 {
        model.aux.mixing.conv_bdy_loc.clear();
        model.aux.mixing.conv_bdy_q.clear();
        model.aux.mixing.top_conv_bdy.clear();
        model.aux.mixing.burn_h_conv_region.clear();
        model.aux.mixing.burn_he_conv_region.clear();
        model.aux.mixing.burn_z_conv_region.clear();
        return Ok(0);
    }
    let (eps_h, eps_he, eps_z) = extract_burn_eps(model)?;
    set_cz_boundary_info(model)?;
    locate_convection_boundaries_in_place(model, &eps_h, &eps_he, &eps_z);
    Ok(model.aux.mixing.conv_bdy_loc.len())
}

/// Computes the Lagrangian face mixing coupling factors $\sigma_k$ (`sig`)
/// across all faces `0..nz` (`get_convection_sigmas` in
/// `star/private/mix_info.f90:1384-1558`), including high-$T_c$ boundary
/// tapering (`sig_min_factor_for_high_Tcenter`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] where upstream calls `mesa_error` on
/// inconsistent mixing-boundary flags (`mix_info.f90:1474, 1482, 1488`).
pub fn compute_convection_sigmas(
    model: &mut StarModel,
    dt: f64,
    mix_factor: f64,
    sig_term_limit: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    if nz == 0 {
        return Ok(());
    }
    if model.aux.mixing.sig_raw.len() != nz {
        model.aux.mixing.resize(nz);
    }
    model.sig[0] = 0.0;
    model.aux.mixing.sig_raw.fill(0.0);
    if mix_factor <= 0.0 {
        model.sig.fill(0.0);
        return Ok(());
    }

    let xmstar = model.xmstar;
    for k in 1..nz {
        let d = model.d_mix[k];
        if d == 0.0 {
            model.sig[k] = 0.0;
            continue;
        }
        let rho_face = (model.dq[k - 1] * model.rho[k]
            + model.dq[k] * model.rho[k - 1])
            / (model.dq[k - 1] + model.dq[k]);
        let f1 = FOUR_PI * model.r[k] * model.r[k] * rho_face;
        let f = f1 * f1;
        let cdc = d * f;
        let cdcterm = mix_factor * cdc;
        let dmavg = xmstar * (model.dq[k] + model.dq[k - 1]) / 2.0;
        let sig_k = cdcterm / dmavg;
        model.sig[k] = if sig_k.is_finite() { sig_k } else { 1.0e99 };
    }

    let max_sig = model
        .sig
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_sig < 1.0 {
        return Ok(());
    }

    for k in 0..nz {
        model.aux.mixing.sig_raw[k] = model.sig[k];
        if model.sig[k] == 0.0 {
            continue;
        }
        if dt > 0.0 && sig_term_limit > 0.0 {
            let siglim = if k > 0 {
                sig_term_limit * xmstar * model.dq[k].min(model.dq[k - 1]) / dt
            } else {
                sig_term_limit * xmstar * model.dq[k] / dt
            };
            if model.sig[k] > siglim {
                model.sig[k] = siglim;
            }
        }
    }

    // High-Tc convective-boundary sigma tapering (`mix_info.f90:1447-1555`).
    let mw = &model.controls.mix_wind;
    let tc = model.t[nz - 1];
    let full_off = mw.tcenter_max_for_sig_min_factor_full_off;
    if tc <= full_off || dt <= 0.0 {
        return Ok(());
    }
    let full_on = mw.tcenter_min_for_sig_min_factor_full_on;
    let mut limit = mw.sig_min_factor_for_high_tcenter;
    if tc < full_on {
        let alfa = (tc - full_off) / (full_on - full_off);
        limit = limit * alfa + (1.0 - alfa);
    }
    let n_bdy = model.aux.mixing.mix_bdy_loc.len();
    if limit >= 1.0 || n_bdy == 0 {
        return Ok(());
    }

    // Boundaries run from centre to surface (`mix_info.f90:1460-1491`).
    // Upstream aborts on inconsistent boundary flags; needing a boundary past
    // the last one, which upstream would read out of bounds, is the same error.
    let bad = |which: u8| {
        StarError::InvalidInput(format!(
            "get_convection_sigmas: bad mix bdy info {which}"
        ))
    };
    let (mut kbot, mut qbot, mut bdy) = if model.aux.mixing.top_mix_bdy[0] {
        (nz - 1, 0.0_f64, 0_usize)
    } else {
        let kb = model.aux.mixing.mix_bdy_loc[0];
        (kb, model.q[kb], 1_usize)
    };
    if bdy >= n_bdy || !model.aux.mixing.top_mix_bdy[bdy] {
        return Err(bad(1));
    }
    let mut ktop = model.aux.mixing.mix_bdy_loc[bdy];
    let mut qtop = model.q[ktop];
    taper_sig_region(model, dt, limit, ktop, kbot, qtop, qbot);

    while bdy + 1 < n_bdy {
        bdy += 1;
        if model.aux.mixing.top_mix_bdy[bdy] {
            return Err(bad(2));
        }
        kbot = model.aux.mixing.mix_bdy_loc[bdy];
        qbot = model.q[kbot];
        bdy += 1;
        if bdy >= n_bdy || !model.aux.mixing.top_mix_bdy[bdy] {
            return Err(bad(3));
        }
        ktop = model.aux.mixing.mix_bdy_loc[bdy];
        qtop = model.q[ktop];
        taper_sig_region(model, dt, limit, ktop, kbot, qtop, qbot);
    }
    Ok(())
}

fn taper_sig_region(
    model: &mut StarModel,
    dt: f64,
    limit: f64,
    ktop: usize,
    kbot: usize,
    qtop: f64,
    qbot: f64,
) {
    let mw = &model.controls.mix_wind;
    let delta_m = model.m[ktop] - model.m[kbot];
    let delta_m_upper = mw.delta_m_upper_for_sig_min_factor;
    let delta_m_lower = mw.delta_m_lower_for_sig_min_factor;
    let max_lim = if delta_m >= delta_m_upper {
        1.0
    } else if delta_m <= delta_m_lower {
        limit
    } else {
        let alfa = (delta_m - delta_m_lower) / (delta_m_upper - delta_m_lower);
        let beta = 1.0 - alfa;
        alfa + beta * limit
    };

    let max_delta_m_to_bdy = mw.max_delta_m_to_bdy_for_sig_min_factor;
    let species = model.species;
    let cushion = 10.0_f64;

    for k in ktop.max(1)..=kbot {
        let mut siglim = model.sig[k];
        if siglim == 0.0 {
            continue;
        }
        let delta_m_to_bdy =
            (model.q[k] - qbot).min(qtop - model.q[k]) * model.xmstar / M_SUN;
        if delta_m_to_bdy >= max_delta_m_to_bdy {
            continue;
        }
        let lim =
            limit + (max_lim - limit) * delta_m_to_bdy / max_delta_m_to_bdy;
        if lim >= 1.0 {
            continue;
        }
        for j in 0..species {
            let xm1 = model.xa[(k - 1) * species + j];
            let x00 = model.xa[k * species + j];
            let (x_val, dx, dm) = if xm1 > x00 {
                (xm1, xm1 - x00, model.dm[k - 1])
            } else {
                // Verbatim `star/private/mix_info.f90:1540-1544` (`dX = -dX`
                // after `dX = x00 - xm1`; tracked under #169 / #268).
                (x00, -(x00 - xm1), model.dm[k])
            };
            if x_val < 1.0e-5 {
                continue;
            }
            if cushion * dt * dx * siglim > dm * x_val {
                siglim = dm * x_val / (dt * dx * cushion);
            }
        }
        if siglim > lim * model.sig[k] {
            model.sig[k] = siglim;
        } else {
            model.sig[k] *= lim;
        }
    }
}

/// Evaluates the explicit mass-conserving diffusive species mixing rate
/// $(dX_{j,k}/dt)_{\mathrm{mix}}$ (`set_dxdt_mix` in `mix_info.f90:2303-2364`).
///
/// Returns a flat array of length `nz * species` in zone-major layout.
#[must_use]
pub fn eval_dxdt_mix(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    let species = model.species;
    let mut dxdt_mix = vec![0.0; nz * species];
    for k in 0..nz {
        let dm_k = model.dm[k].max(1.0e-99);
        let sig_00 = if k > 0 { model.sig[k] } else { 0.0 };
        let sig_p1 = if k + 1 < nz { model.sig[k + 1] } else { 0.0 };
        for j in 0..species {
            let x_00 = model.xa[k * species + j];
            let flux_in = if k > 0 {
                sig_00 * (model.xa[(k - 1) * species + j] - x_00)
            } else {
                0.0
            };
            let flux_out = if k + 1 < nz {
                sig_p1 * (x_00 - model.xa[(k + 1) * species + j])
            } else {
                0.0
            };
            dxdt_mix[k * species + j] = (flux_in - flux_out) / dm_k;
        }
    }
    dxdt_mix
}
