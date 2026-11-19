// SPDX-License-Identifier: LGPL-3.0-or-later
//! Complete `&eos` and `&kap` namelist population and validation
//! (`eos/private/eos_ctrls_io.f90`, `kap/private/kap_ctrls_io.f90`, and
//! `kap/private/kap_aesopus.f90:91-110`, `#161` `F2-1`).

use std::path::Path;

use flexmesa_eos::SkyeSolidMixingRule;
use flexmesa_kap::{KapCoOption, KapLowTOption, KapOption};
use flexmesa_schema::ControlValue;

use crate::controls::StarError;
use crate::inlist::{InlistBundle, SectionControls};

const KAP_MAX_DIM: usize = 50;

fn extract_f64(val: &ControlValue) -> Result<f64, String> {
    match val {
        ControlValue::Float(f) if f.is_finite() => Ok(*f),
        ControlValue::Float(f) => Err(format!("non-finite float {f}")),
        ControlValue::Integer(i) => Ok(*i as f64),
        other => Err(format!("expected real number, found {other:?}")),
    }
}

fn extract_i64(val: &ControlValue) -> Result<i64, String> {
    match val {
        ControlValue::Integer(i) => Ok(*i),
        other => Err(format!("expected integer, found {other:?}")),
    }
}

fn extract_bool(val: &ControlValue) -> Result<bool, String> {
    match val {
        ControlValue::Logical(b) => Ok(*b),
        other => Err(format!("expected boolean, found {other:?}")),
    }
}

fn extract_string(val: &ControlValue) -> Result<String, String> {
    match val {
        ControlValue::String(s) => Ok(s.clone()),
        other => Err(format!("expected string, found {other:?}")),
    }
}

fn apply_micro_indexed_slice<T: Clone>(
    sec_name: &str,
    sec: &SectionControls,
    key: &str,
    allow_scalar_as_first: bool,
    dst: &mut [T],
    extract: impl Fn(&ControlValue) -> Result<T, String>,
) -> Result<(), StarError> {
    let low = key.to_ascii_lowercase();
    if let Some(val) = sec.scalars.get(&low) {
        if !allow_scalar_as_first {
            return Err(StarError::InvalidInput(format!(
                "missing 1-based subscript '(1..={})' for array control '&{sec_name}::{low}'",
                dst.len()
            )));
        }
        let v = extract(val).map_err(|msg| {
            StarError::InvalidInput(format!(
                "invalid value for '&{sec_name}::{low}': {msg}"
            ))
        })?;
        if let Some(slot) = dst.first_mut() {
            *slot = v;
        }
    }
    if let Some(list) = sec.indexed.get(&low) {
        for (sub, val) in list {
            let trimmed = sub.trim();
            let v = extract(val).map_err(|msg| {
                StarError::InvalidInput(format!(
                    "invalid value for '&{sec_name}::{low}({trimmed})': {msg}"
                ))
            })?;
            if trimmed == ":" {
                dst.fill(v);
                continue;
            }
            let idx = trimmed.parse::<usize>().map_err(|_| {
                StarError::InvalidInput(format!(
                    "invalid subscript '({sub})' for '&{sec_name}::{low}'"
                ))
            })?;
            if idx == 0 || idx > dst.len() {
                return Err(StarError::InvalidInput(format!(
                    "subscript '({idx})' out of 1-based bounds (1..={}) for '&{sec_name}::{low}'",
                    dst.len()
                )));
            }
            dst[idx - 1] = v;
        }
    }
    Ok(())
}

fn reject_true_hook(
    sec_name: &str,
    sec: &SectionControls,
    key: &str,
) -> Result<(), StarError> {
    if sec.try_get_bool(sec_name, key)?.unwrap_or(false) {
        return Err(StarError::InvalidInput(format!(
            "unsupported '&{sec_name}::{key}' = .true.: custom Fortran physics hooks are not supported"
        )));
    }
    Ok(())
}

fn validate_terminal_debug_bounds(
    sec_name: &str,
    sec: &SectionControls,
) -> Result<(), StarError> {
    let _ = sec.try_get_bool(sec_name, "dbg")?;
    for key in [
        "logt_lo",
        "logt_hi",
        "logrho_lo",
        "logrho_hi",
        "x_lo",
        "x_hi",
        "z_lo",
        "z_hi",
    ] {
        let _ = sec.try_get_f64(sec_name, key)?;
    }
    Ok(())
}

impl InlistBundle {
    pub(crate) fn populate_eos_and_kap_configs(
        &mut self,
    ) -> Result<(), StarError> {
        self.populate_eos_section()?;
        self.populate_kap_section()
    }

    #[allow(clippy::too_many_lines)]
    fn populate_eos_section(&mut self) -> Result<(), StarError> {
        let Some(eos) = self.sections.get("eos") else {
            return Ok(());
        };
        validate_terminal_debug_bounds("eos", eos)?;
        reject_true_hook("eos", eos, "use_other_eos_component")?;
        reject_true_hook("eos", eos, "use_other_eos_results")?;

        let cfg = &mut self.eos_config;
        eos.try_bool_into("eos", &mut cfg.use_free_eos, "use_freeeos")?;
        eos.try_bool_into("eos", &mut cfg.use_opal_scvh, "use_opal_scvh")?;
        eos.try_bool_into("eos", &mut cfg.use_pc, "use_pc")?;
        eos.try_bool_into("eos", &mut cfg.use_skye, "use_skye")?;
        eos.try_bool_into("eos", &mut cfg.use_cms, "use_cms")?;
        eos.try_bool_into(
            "eos",
            &mut cfg.okay_to_convert_ierr_to_skip,
            "okay_to_convert_ierr_to_skip",
        )?;
        eos.try_bool_into(
            "eos",
            &mut cfg.eosdt_use_linear_interp_to_helm,
            "eosdt_use_linear_interp_to_helm",
        )?;

        if let Some(inc_rad) = eos.try_get_bool("eos", "include_radiation")? {
            cfg.helm.include_radiation = inc_rad;
            cfg.pc.include_radiation = inc_rad;
            cfg.cms.include_radiation = inc_rad;
        }
        eos.try_bool_into(
            "eos",
            &mut cfg.helm.include_elec_pos,
            "include_elec_pos",
        )?;
        eos.try_f64_into(
            "eos",
            &mut cfg.helm.coulomb_temp_cut,
            "coulomb_temp_cut_helm",
        )?;
        eos.try_f64_into(
            "eos",
            &mut cfg.helm.coulomb_den_cut,
            "coulomb_den_cut_helm",
        )?;

        if let Some(lin_x) =
            eos.try_get_bool("eos", "eosdt_use_linear_interp_for_x")?
        {
            cfg.free_eos.use_linear_interp_for_x = lin_x;
            cfg.opal_scvh.use_linear_interp_for_x = lin_x;
        }
        if let Some(fuzz) = eos.try_get_f64("eos", "tiny_fuzz")? {
            cfg.free_eos.tiny_fuzz = fuzz;
            cfg.opal_scvh.tiny_fuzz = fuzz;
        }
        if let Some(prefix) = eos.try_get_str("eos", "eosdt_file_prefix")? {
            let trimmed = prefix.trim();
            if trimmed != "mesa" {
                return Err(StarError::InvalidInput(format!(
                    "unsupported '&eos::eosdt_file_prefix' = '{trimmed}': only bundled 'mesa' tables are supported"
                )));
            }
        }

        // OPAL/SCVH controls
        for (dst, key) in [
            (&mut cfg.opal_scvh.log_t_low_all_helm, "logt_low_all_helm"),
            (&mut cfg.opal_scvh.log_t_low_all_scvh, "logt_low_all_scvh"),
            (&mut cfg.opal_scvh.log_t_all_opal, "logt_all_opal"),
            (&mut cfg.opal_scvh.log_t_all_helm, "logt_all_helm"),
            (
                &mut cfg.opal_scvh.log_rho1_opal_scvh_limit,
                "logrho1_opal_scvh_limit",
            ),
            (
                &mut cfg.opal_scvh.log_rho2_opal_scvh_limit,
                "logrho2_opal_scvh_limit",
            ),
            (
                &mut cfg.opal_scvh.log_rho_min_opal_scvh_limit,
                "logrho_min_opal_scvh_limit",
            ),
            (&mut cfg.opal_scvh.log_q_max_opal_scvh, "logq_max_opal_scvh"),
            (&mut cfg.opal_scvh.log_q_min_opal_scvh, "logq_min_opal_scvh"),
            (&mut cfg.opal_scvh.z_all_opal, "z_all_opal"),
            (&mut cfg.opal_scvh.z_all_helm, "z_all_helm"),
        ] {
            eos.try_f64_into("eos", dst, key)?;
        }

        // FreeEOS controls
        for (dst, key) in [
            (&mut cfg.free_eos.log_q_max_hi, "logq_max_freeeos_hi"),
            (&mut cfg.free_eos.log_q_max_lo, "logq_max_freeeos_lo"),
            (&mut cfg.free_eos.log_q_min_hi, "logq_min_freeeos_hi"),
            (&mut cfg.free_eos.log_q_min_lo, "logq_min_freeeos_lo"),
            (&mut cfg.free_eos.log_rho_min_hi, "logrho_min_freeeos_hi"),
            (&mut cfg.free_eos.log_rho_min_lo, "logrho_min_freeeos_lo"),
            (&mut cfg.free_eos.log_rho_max_hi, "logrho_max_freeeos_hi"),
            (&mut cfg.free_eos.log_rho_max_lo, "logrho_max_freeeos_lo"),
            (&mut cfg.free_eos.log_t_min_hi, "logt_min_freeeos_hi"),
            (&mut cfg.free_eos.log_t_min_lo, "logt_min_freeeos_lo"),
            (&mut cfg.free_eos.log_t_max_hi, "logt_max_freeeos_hi"),
            (&mut cfg.free_eos.log_t_max_lo, "logt_max_freeeos_lo"),
            (
                &mut cfg.free_eos.log_q_cut_lo_z_max,
                "logq_cut_freeeos_lo_z_max",
            ),
            (
                &mut cfg.free_eos.log_q_cut_lo_z_hi,
                "logq_cut_lo_z_freeeos_hi",
            ),
            (
                &mut cfg.free_eos.log_q_cut_lo_z_lo,
                "logq_cut_lo_z_freeeos_lo",
            ),
            (
                &mut cfg.free_eos.log_q_cut_hi_z_hi,
                "logq_cut_hi_z_freeeos_hi",
            ),
            (
                &mut cfg.free_eos.log_q_cut_hi_z_lo,
                "logq_cut_hi_z_freeeos_lo",
            ),
            (&mut cfg.free_eos.log_rho_cut_hi, "logrho_cut_freeeos_hi"),
            (&mut cfg.free_eos.log_rho_cut_lo, "logrho_cut_freeeos_lo"),
            (&mut cfg.free_eos.log_t_cut_hi, "logt_cut_freeeos_hi"),
            (&mut cfg.free_eos.log_t_cut_lo, "logt_cut_freeeos_lo"),
        ] {
            eos.try_f64_into("eos", dst, key)?;
        }

        let mut suffixes: [String; 15] = std::array::from_fn(|idx| {
            if idx >= 12 {
                String::from("_CO")
            } else {
                String::new()
            }
        });
        apply_micro_indexed_slice(
            "eos",
            eos,
            "suffix_for_freeeos_z",
            false,
            &mut suffixes,
            extract_string,
        )?;
        for (idx, s) in suffixes.iter().enumerate() {
            let expected = if idx >= 12 { "_CO" } else { "" };
            if s.trim() != expected {
                return Err(StarError::InvalidInput(format!(
                    "unsupported '&eos::suffix_for_freeeos_z({})' = '{}': only bundled FreeEOS tables ('{expected}') are supported",
                    idx + 1,
                    s.trim()
                )));
            }
        }

        // CMS controls
        eos.try_bool_into(
            "eos",
            &mut cfg.cms.use_fixed_composition,
            "cms_use_fixed_composition",
        )?;
        if let Some(idx) =
            eos.try_get_i64("eos", "cms_fixed_composition_index")?
        {
            cfg.cms.fixed_composition_index = idx as isize;
        }
        if cfg.use_cms
            && cfg.cms.use_fixed_composition
            && !(0..=10).contains(&cfg.cms.fixed_composition_index)
        {
            return Err(StarError::InvalidInput(format!(
                "invalid '&eos::cms_fixed_composition_index' = {}: expected 0..=10 when cms_use_fixed_composition is .true.",
                cfg.cms.fixed_composition_index
            )));
        }
        for (dst, key) in [
            (&mut cfg.cms.max_z_for_any_cms, "max_z_for_any_cms"),
            (&mut cfg.cms.max_z_for_all_cms, "max_z_for_all_cms"),
            (&mut cfg.cms.logq_max_for_any_cms, "logq_max_for_any_cms"),
            (&mut cfg.cms.logq_max_for_all_cms, "logq_max_for_all_cms"),
            (&mut cfg.cms.logq_min_for_any_cms, "logq_min_for_any_cms"),
            (&mut cfg.cms.logq_min_for_all_cms, "logq_min_for_all_cms"),
            (
                &mut cfg.cms.logrho_max_for_all_cms,
                "logrho_max_for_all_cms",
            ),
            (
                &mut cfg.cms.logrho_max_for_any_cms,
                "logrho_max_for_any_cms",
            ),
            (
                &mut cfg.cms.logrho_min_for_all_cms,
                "logrho_min_for_all_cms",
            ),
            (
                &mut cfg.cms.logrho_min_for_any_cms,
                "logrho_min_for_any_cms",
            ),
            (&mut cfg.cms.logt_max_for_all_cms, "logt_max_for_all_cms"),
            (&mut cfg.cms.logt_max_for_any_cms, "logt_max_for_any_cms"),
            (&mut cfg.cms.logt_min_for_all_cms, "logt_min_for_all_cms"),
            (&mut cfg.cms.logt_min_for_any_cms, "logt_min_for_any_cms"),
            (
                &mut cfg.cms.logt_max_for_all_cms_pure_he,
                "logt_max_for_all_cms_pure_he",
            ),
            (
                &mut cfg.cms.logt_max_for_any_cms_pure_he,
                "logt_max_for_any_cms_pure_he",
            ),
        ] {
            eos.try_f64_into("eos", dst, key)?;
        }

        // PC controls
        eos.try_bool_into(
            "eos",
            &mut cfg.pc.pc_use_gamma_limit_instead_of_t,
            "pc_use_gamma_limit_instead_of_t",
        )?;
        for (dst, key) in [
            (
                &mut cfg.pc.mass_fraction_limit_for_pc,
                "mass_fraction_limit_for_pc",
            ),
            (&mut cfg.pc.log_rho1_pc_limit, "logrho1_pc_limit"),
            (&mut cfg.pc.log_rho2_pc_limit, "logrho2_pc_limit"),
            (&mut cfg.pc.log_t1_pc_limit, "logt1_pc_limit"),
            (&mut cfg.pc.log_t2_pc_limit, "logt2_pc_limit"),
            (&mut cfg.pc.log_gamma_e_all_helm, "log_gamma_e_all_helm"),
            (&mut cfg.pc.log_gamma_e_all_pc, "log_gamma_e_all_pc"),
            (&mut cfg.pc.pc_gamma_start_crystal, "pc_gamma_start_crystal"),
            (&mut cfg.pc.pc_gamma_full_crystal, "pc_gamma_full_crystal"),
        ] {
            eos.try_f64_into("eos", dst, key)?;
        }

        // Skye controls
        eos.try_bool_into(
            "eos",
            &mut cfg.skye.skye_use_ion_offsets,
            "skye_use_ion_offsets",
        )?;
        eos.try_bool_into(
            "eos",
            &mut cfg.skye.use_simple_skye_blends,
            "use_simple_skye_blends",
        )?;
        if let Some(rule) = eos.try_get_str("eos", "skye_solid_mixing_rule")? {
            cfg.skye.skye_solid_mixing_rule = match rule.trim() {
                "PC" => SkyeSolidMixingRule::Pc,
                "Ogata" => SkyeSolidMixingRule::Ogata,
                other => {
                    return Err(StarError::InvalidInput(format!(
                        "invalid '&eos::skye_solid_mixing_rule' = '{other}': expected 'PC' or 'Ogata'"
                    )));
                }
            };
        }
        for (dst, key) in [
            (
                &mut cfg.skye.mass_fraction_limit_for_skye,
                "mass_fraction_limit_for_skye",
            ),
            (
                &mut cfg.skye.skye_min_gamma_for_solid,
                "skye_min_gamma_for_solid",
            ),
            (
                &mut cfg.skye.skye_max_gamma_for_liquid,
                "skye_max_gamma_for_liquid",
            ),
            (
                &mut cfg.skye.log_rho_min_for_any_skye,
                "logrho_min_for_any_skye",
            ),
            (
                &mut cfg.skye.log_rho_min_for_all_skye,
                "logrho_min_for_all_skye",
            ),
            (
                &mut cfg.skye.log_t_min_for_any_skye,
                "logt_min_for_any_skye",
            ),
            (
                &mut cfg.skye.log_t_min_for_all_skye,
                "logt_min_for_all_skye",
            ),
        ] {
            eos.try_f64_into("eos", dst, key)?;
        }

        // User controls (`eos_*_ctrl(1..=10)`)
        apply_micro_indexed_slice(
            "eos",
            eos,
            "eos_ctrl",
            false,
            &mut cfg.eos_ctrl,
            extract_f64,
        )?;
        apply_micro_indexed_slice(
            "eos",
            eos,
            "eos_integer_ctrl",
            false,
            &mut cfg.eos_integer_ctrl,
            extract_i64,
        )?;
        apply_micro_indexed_slice(
            "eos",
            eos,
            "eos_logical_ctrl",
            false,
            &mut cfg.eos_logical_ctrl,
            extract_bool,
        )?;
        let mut eos_char_ctrl: [String; 10] =
            std::array::from_fn(|_| String::new());
        apply_micro_indexed_slice(
            "eos",
            eos,
            "eos_character_ctrl",
            false,
            &mut eos_char_ctrl,
            extract_string,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn populate_kap_section(&mut self) -> Result<(), StarError> {
        let Some(kap) = self.sections.get("kap") else {
            return Ok(());
        };
        let _ = kap.try_get_bool("kap", "show_info")?;
        validate_terminal_debug_bounds("kap", kap)?;
        reject_true_hook("kap", kap, "use_other_elect_cond_opacity")?;
        reject_true_hook("kap", kap, "use_other_compton_opacity")?;
        reject_true_hook("kap", kap, "use_other_radiative_opacity")?;

        // Validate user custom-table dimension/grid arrays (`kap_ctrls_io.f90:38-55`).
        // Upstream only activates custom tables when `kap_*_prefix` is unrecognized,
        // which `Kap*Option::from_prefix` rejects loudly below.
        for num_key in [
            "user_num_kap_xs",
            "user_num_kap_zs",
            "user_num_kap_co_xs",
            "user_num_kap_co_zs",
            "user_num_kap_lowt_xs",
            "user_num_kap_lowt_zs",
        ] {
            if let Some(n) = kap.try_get_i64("kap", num_key)? {
                if n < 0 || (n as usize) > KAP_MAX_DIM {
                    return Err(StarError::InvalidInput(format!(
                        "invalid '&kap::{num_key}' = {n}: expected 0..={KAP_MAX_DIM}"
                    )));
                }
            }
        }
        let mut user_i64_buf = [0_i64; KAP_MAX_DIM];
        for int_arr_key in [
            "user_num_kap_xs_for_this_z",
            "user_num_kap_co_xs_for_this_z",
            "user_num_kap_lowt_xs_for_this_z",
        ] {
            apply_micro_indexed_slice(
                "kap",
                kap,
                int_arr_key,
                true,
                &mut user_i64_buf,
                extract_i64,
            )?;
        }
        let mut user_f64_buf = [0.0_f64; KAP_MAX_DIM];
        for real_arr_key in [
            "user_kap_xs",
            "user_kap_zs",
            "user_kap_co_xs",
            "user_kap_co_zs",
            "user_kap_lowt_xs",
            "user_kap_lowt_zs",
        ] {
            apply_micro_indexed_slice(
                "kap",
                kap,
                real_arr_key,
                true,
                &mut user_f64_buf,
                extract_f64,
            )?;
        }

        let kcfg = &mut self.kap_config;
        if let Some(v) = kap.try_get_f64("kap", "zbase")? {
            kcfg.zbase = v;
            self.controls.zbase = v;
        }
        if let Some(s) = kap.try_get_str("kap", "kap_file_prefix")? {
            kcfg.kap_option = KapOption::from_prefix(s).map_err(|e| {
                StarError::InvalidInput(format!(
                    "invalid '&kap::kap_file_prefix' = '{s}': {e}"
                ))
            })?;
        }
        if let Some(s) = kap.try_get_str("kap", "kap_co_prefix")? {
            kcfg.kap_co_option = KapCoOption::from_prefix(s).map_err(|e| {
                StarError::InvalidInput(format!(
                    "invalid '&kap::kap_co_prefix' = '{s}': {e}"
                ))
            })?;
        }
        if let Some(s) = kap.try_get_str("kap", "kap_lowt_prefix")? {
            kcfg.kap_low_t_option =
                KapLowTOption::from_prefix(s).map_err(|e| {
                    StarError::InvalidInput(format!(
                        "invalid '&kap::kap_lowt_prefix' = '{s}': {e}"
                    ))
                })?;
        }
        if let Some(s) = kap.try_get_str("kap", "aesopus_filename")? {
            kcfg.aesopus_filename = s.trim().to_owned();
        }
        if kcfg.kap_low_t_option == KapLowTOption::Aesopus {
            let fname = kcfg.aesopus_filename.trim();
            if fname.is_empty() {
                return Err(StarError::InvalidInput(String::from(
                    "failed to specify '&kap::aesopus_filename' when kap_lowT_prefix = 'AESOPUS'",
                )));
            }
            if Path::new(fname).file_name().and_then(|s| s.to_str())
                != Some("AESOPUS_AGSS09.h5")
            {
                return Err(StarError::InvalidInput(format!(
                    "unsupported '&kap::aesopus_filename' = '{fname}': only bundled 'AESOPUS_AGSS09.h5' is supported"
                )));
            }
        }

        if let Some(v) = kap.try_get_f64("kap", "kap_blend_logt_upper_bdy")? {
            if v > 0.0 {
                kcfg.kap_blend_log_t_upper_bdy = v;
            }
        }
        if let Some(v) = kap.try_get_f64("kap", "kap_blend_logt_lower_bdy")? {
            if v > 0.0 {
                kcfg.kap_blend_log_t_lower_bdy = v;
            }
        }

        for (dst, key) in [
            (
                &mut kcfg.cubic_interpolation_in_x,
                "cubic_interpolation_in_x",
            ),
            (
                &mut kcfg.cubic_interpolation_in_z,
                "cubic_interpolation_in_z",
            ),
            (
                &mut kcfg.include_electron_conduction,
                "include_electron_conduction",
            ),
            (
                &mut kcfg.use_blouin_conductive_opacities,
                "use_blouin_conductive_opacities",
            ),
            (&mut kcfg.use_zbase_for_type1, "use_zbase_for_type1"),
            (&mut kcfg.use_type2_opacities, "use_type2_opacities"),
        ] {
            kap.try_bool_into("kap", dst, key)?;
        }

        for (dst, key) in [
            (&mut kcfg.kap_type2_full_off_x, "kap_type2_full_off_x"),
            (&mut kcfg.kap_type2_full_on_x, "kap_type2_full_on_x"),
            (&mut kcfg.kap_type2_full_off_dz, "kap_type2_full_off_dz"),
            (&mut kcfg.kap_type2_full_on_dz, "kap_type2_full_on_dz"),
        ] {
            kap.try_f64_into("kap", dst, key)?;
        }
        if kcfg.kap_type2_full_off_x > 0.71 {
            return Err(StarError::InvalidInput(format!(
                "'&kap::kap_type2_full_off_x' ({}) must be <= 0.71",
                kcfg.kap_type2_full_off_x
            )));
        }
        if kcfg.kap_type2_full_on_x > 0.71 {
            return Err(StarError::InvalidInput(format!(
                "'&kap::kap_type2_full_on_x' ({}) must be <= 0.71",
                kcfg.kap_type2_full_on_x
            )));
        }
        if kcfg.kap_type2_full_off_x < kcfg.kap_type2_full_on_x {
            return Err(StarError::InvalidInput(format!(
                "'&kap::kap_type2_full_off_x' ({}) must be >= '&kap::kap_type2_full_on_x' ({})",
                kcfg.kap_type2_full_off_x, kcfg.kap_type2_full_on_x
            )));
        }
        if kcfg.kap_type2_full_off_dz > kcfg.kap_type2_full_on_dz {
            return Err(StarError::InvalidInput(format!(
                "'&kap::kap_type2_full_off_dz' ({}) must be <= '&kap::kap_type2_full_on_dz' ({})",
                kcfg.kap_type2_full_off_dz, kcfg.kap_type2_full_on_dz
            )));
        }

        apply_micro_indexed_slice(
            "kap",
            kap,
            "kap_ctrl",
            false,
            &mut kcfg.kap_ctrl,
            extract_f64,
        )?;
        apply_micro_indexed_slice(
            "kap",
            kap,
            "kap_integer_ctrl",
            false,
            &mut kcfg.kap_integer_ctrl,
            extract_i64,
        )?;
        apply_micro_indexed_slice(
            "kap",
            kap,
            "kap_logical_ctrl",
            false,
            &mut kcfg.kap_logical_ctrl,
            extract_bool,
        )?;
        apply_micro_indexed_slice(
            "kap",
            kap,
            "kap_character_ctrl",
            false,
            &mut kcfg.kap_character_ctrl,
            extract_string,
        )?;
        Ok(())
    }
}
