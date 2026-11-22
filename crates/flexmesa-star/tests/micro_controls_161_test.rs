// SPDX-License-Identifier: LGPL-3.0-or-later
//! Issue `#161` (`F2-1`) `&eos` and `&kap` namelist coverage, population, and
//! fail-loudly validation tests.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_eos::SkyeSolidMixingRule;
use flexmesa_kap::{KapCoOption, KapLowTOption, KapOption};
use flexmesa_schema::{is_include_key, upstream_section_keys};
use flexmesa_star::inlist::InlistBundle;
use flexmesa_star::inlist_inventory::supported_star_control_inventory;
use flexmesa_star::StarError;

struct TempCaseGuard {
    root: PathBuf,
}

impl TempCaseGuard {
    fn new(tag: &str) -> Self {
        static SEQ: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "flexmesa_star_161_micro_ctrls_{tag}_{}_{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp case dir");
        Self { root }
    }

    fn load_inlist(
        &self,
        name: &str,
        contents: &str,
    ) -> Result<InlistBundle, StarError> {
        let path = self.root.join(name);
        fs::write(&path, contents).expect("write inlist");
        InlistBundle::from_file(&path)
    }
}

impl Drop for TempCaseGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn test_eos_and_kap_inventories_cover_entire_mesa_26_4_1_schema() {
    let inv = supported_star_control_inventory();
    assert_eq!(inv.eos_supported.len(), 91);
    assert_eq!(inv.eos_terminal_output.len(), 9);
    let eos_upstream = upstream_section_keys("eos").expect("eos schema");
    assert!(eos_upstream.len() >= 93);
    for &key in eos_upstream {
        let covered = inv.eos_supported.contains(key)
            || inv.eos_terminal_output.contains(key)
            || is_include_key("eos", key);
        assert!(covered, "upstream &eos key '{key}' is not covered");
    }

    assert_eq!(inv.kap_supported.len(), 39);
    assert_eq!(inv.kap_terminal_output.len(), 10);
    let kap_upstream = upstream_section_keys("kap").expect("kap schema");
    assert!(kap_upstream.len() >= 51);
    for &key in kap_upstream {
        let covered = inv.kap_supported.contains(key)
            || inv.kap_terminal_output.contains(key)
            || is_include_key("kap", key);
        assert!(covered, "upstream &kap key '{key}' is not covered");
    }
}

#[test]
fn test_eos_namelist_populates_all_component_configs_and_terminal_keys() {
    let tmp = TempCaseGuard::new("eos_populate");
    let bundle = tmp
        .load_inlist(
            "inlist_eos",
            "&eos\n\
               use_FreeEOS = .false.\n\
               use_OPAL_SCVH = .true.\n\
               use_PC = .true.\n\
               use_Skye = .true.\n\
               use_CMS = .true.\n\
               okay_to_convert_ierr_to_skip = .false.\n\
               eosDT_use_linear_interp_to_HELM = .true.\n\
               eosDT_use_linear_interp_for_X = .true.\n\
               eosDT_file_prefix = 'mesa'\n\
               tiny_fuzz = 2.5d-6\n\
               include_radiation = .false.\n\
               include_elec_pos = .false.\n\
               coulomb_temp_cut_HELM = -2.0d0\n\
               coulomb_den_cut_HELM = -35.0d0\n\
               logT_low_all_HELM = 2.25d0\n\
               logT_all_OPAL = 7.55d0\n\
               logQ_max_FreeEOS_hi = 5.55d0\n\
               logT_cut_FreeEOS_lo = 3.65d0\n\
               suffix_for_FreeEOS_Z(1) = ''\n\
               suffix_for_FreeEOS_Z(13) = '_CO'\n\
               CMS_use_fixed_composition = .true.\n\
               CMS_fixed_composition_index = 4\n\
               logT_max_for_all_CMS_pure_He = 6.1d0\n\
               logT_max_for_any_CMS_pure_He = 6.3d0\n\
               PC_use_Gamma_limit_instead_of_T = .true.\n\
               mass_fraction_limit_for_PC = 1.0d-4\n\
               Skye_use_ion_offsets = .true.\n\
               use_simple_Skye_blends = .true.\n\
               Skye_solid_mixing_rule = 'Ogata'\n\
               mass_fraction_limit_for_Skye = 1.0d-8\n\
               eos_ctrl(:) = 1.5d0\n\
               eos_ctrl(3) = -4.25d0\n\
               eos_integer_ctrl(2) = 42\n\
               eos_logical_ctrl(10) = .true.\n\
               eos_character_ctrl(1) = 'tag'\n\
               dbg = .false.\n\
               logT_lo = 5.1d0\n\
               logT_hi = 5.2d0\n\
             /\n",
        )
        .expect("&eos inlist must parse cleanly");

    let cfg = &bundle.eos_config;
    assert!(!cfg.use_free_eos);
    assert!(cfg.use_opal_scvh);
    assert!(cfg.use_pc);
    assert!(cfg.use_skye);
    assert!(cfg.use_cms);
    assert!(!cfg.okay_to_convert_ierr_to_skip);
    assert!(cfg.eosdt_use_linear_interp_to_helm);
    assert!(cfg.free_eos.use_linear_interp_for_x);
    assert!(cfg.opal_scvh.use_linear_interp_for_x);
    assert_eq!(cfg.free_eos.tiny_fuzz, 2.5e-6);
    assert_eq!(cfg.opal_scvh.tiny_fuzz, 2.5e-6);
    assert!(!cfg.helm.include_radiation);
    assert!(!cfg.pc.include_radiation);
    assert!(!cfg.cms.include_radiation);
    assert!(!cfg.helm.include_elec_pos);
    assert_eq!(cfg.helm.coulomb_temp_cut, -2.0);
    assert_eq!(cfg.helm.coulomb_den_cut, -35.0);
    assert_eq!(cfg.opal_scvh.log_t_low_all_helm, 2.25);
    assert_eq!(cfg.opal_scvh.log_t_all_opal, 7.55);
    assert_eq!(cfg.free_eos.log_q_max_hi, 5.55);
    assert_eq!(cfg.free_eos.log_t_cut_lo, 3.65);
    assert!(cfg.cms.use_fixed_composition);
    assert_eq!(cfg.cms.fixed_composition_index, 4);
    assert_eq!(cfg.cms.logt_max_for_all_cms_pure_he, 6.1);
    assert_eq!(cfg.cms.logt_max_for_any_cms_pure_he, 6.3);
    assert!(cfg.pc.pc_use_gamma_limit_instead_of_t);
    assert_eq!(cfg.pc.mass_fraction_limit_for_pc, 1.0e-4);
    assert!(cfg.skye.skye_use_ion_offsets);
    assert!(cfg.skye.use_simple_skye_blends);
    assert_eq!(cfg.skye.skye_solid_mixing_rule, SkyeSolidMixingRule::Ogata);
    assert_eq!(cfg.skye.mass_fraction_limit_for_skye, 1.0e-8);
    assert_eq!(cfg.eos_ctrl[0], 1.5);
    assert_eq!(cfg.eos_ctrl[2], -4.25);
    assert_eq!(cfg.eos_integer_ctrl[1], 42);
    assert!(cfg.eos_logical_ctrl[9]);
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
}

#[test]
fn test_eos_namelist_rejects_unbundled_tables_hooks_and_invalid_options() {
    let tmp = TempCaseGuard::new("eos_loud");
    for (tag, body, needle) in [
        (
            "hook_comp",
            "&eos\n  use_other_eos_component = .true.\n/\n",
            "use_other_eos_component",
        ),
        (
            "hook_res",
            "&eos\n  use_other_eos_results = .true.\n/\n",
            "use_other_eos_results",
        ),
        (
            "prefix",
            "&eos\n  eosDT_file_prefix = 'custom'\n/\n",
            "eosdt_file_prefix",
        ),
        (
            "freeeos_suffix",
            "&eos\n  suffix_for_FreeEOS_Z(1) = '_alt'\n/\n",
            "suffix_for_freeeos_z(1)",
        ),
        (
            "skye_rule",
            "&eos\n  Skye_solid_mixing_rule = 'Linear'\n/\n",
            "skye_solid_mixing_rule",
        ),
        (
            "cms_idx",
            "&eos\n  use_CMS = .true.\n  CMS_use_fixed_composition = .true.\n  CMS_fixed_composition_index = 12\n/\n",
            "cms_fixed_composition_index",
        ),
    ] {
        let err = tmp
            .load_inlist(&format!("inlist_{tag}"), body)
            .expect_err(tag);
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "{tag}: unexpected variant {err:?}"
        );
        assert!(
            err.to_string().contains(needle),
            "{tag}: expected '{needle}' in '{err}'"
        );
    }
}

#[test]
fn test_kap_namelist_populates_options_blend_bounds_and_low_z_user_arrays() {
    let tmp = TempCaseGuard::new("kap_populate");
    let bundle = tmp
        .load_inlist(
            "inlist_kap",
            "&kap\n\
               Zbase = 0.0001d0\n\
               kap_file_prefix = 'OP_a09_nans_removed_by_hand'\n\
               kap_CO_prefix = 'a09_co'\n\
               kap_lowT_prefix = 'AESOPUS'\n\
               AESOPUS_filename = 'AESOPUS_AGSS09.h5'\n\
               kap_blend_logT_upper_bdy = 4.18d0\n\
               kap_blend_logT_lower_bdy = 4.02d0\n\
               cubic_interpolation_in_X = .false.\n\
               cubic_interpolation_in_Z = .false.\n\
               include_electron_conduction = .false.\n\
               use_blouin_conductive_opacities = .true.\n\
               use_Zbase_for_Type1 = .true.\n\
               use_Type2_opacities = .true.\n\
               kap_Type2_full_off_X = 1.0d-3\n\
               kap_Type2_full_on_X = 1.0d-6\n\
               kap_Type2_full_off_dZ = 1.0d-4\n\
               kap_Type2_full_on_dZ = 1.0d-2\n\
               user_num_kap_Xs = 8\n\
               user_num_kap_Zs = 4\n\
               user_num_kap_Xs_for_this_Z = 8, 8, 8, 8\n\
               user_kap_Xs = 0.0d0, 0.1d0, 0.2d0, 0.35d0, 0.5d0, 0.7d0, 0.8d0, 0.9d0\n\
               user_kap_Zs = 0.0d0, 0.0001d0, 0.0003d0, 0.001d0\n\
               kap_ctrl(:) = 0.25d0\n\
               kap_ctrl(4) = 9.5d0\n\
               kap_integer_ctrl(1) = 7\n\
               kap_logical_ctrl(2) = .true.\n\
               kap_character_ctrl(3) = 'op'\n\
               show_info = .false.\n\
               dbg = .false.\n\
             /\n",
        )
        .expect("&kap inlist must parse cleanly");

    let kcfg = &bundle.kap_config;
    assert_eq!(kcfg.zbase, 1.0e-4);
    assert_eq!(bundle.controls.zbase, 1.0e-4);
    assert_eq!(kcfg.kap_option, KapOption::OpA09);
    assert_eq!(kcfg.kap_co_option, KapCoOption::A09Co);
    assert_eq!(kcfg.kap_low_t_option, KapLowTOption::Aesopus);
    assert_eq!(kcfg.aesopus_filename, "AESOPUS_AGSS09.h5");
    assert_eq!(kcfg.kap_blend_log_t_upper_bdy, 4.18);
    assert_eq!(kcfg.kap_blend_log_t_lower_bdy, 4.02);
    assert!(!kcfg.cubic_interpolation_in_x);
    assert!(!kcfg.cubic_interpolation_in_z);
    assert!(!kcfg.include_electron_conduction);
    assert!(kcfg.use_blouin_conductive_opacities);
    assert!(kcfg.use_zbase_for_type1);
    assert!(kcfg.use_type2_opacities);
    assert_eq!(kcfg.kap_type2_full_off_x, 1.0e-3);
    assert_eq!(kcfg.kap_type2_full_on_x, 1.0e-6);
    assert_eq!(kcfg.kap_type2_full_off_dz, 1.0e-4);
    assert_eq!(kcfg.kap_type2_full_on_dz, 1.0e-2);
    assert_eq!(kcfg.kap_ctrl[0], 0.25);
    assert_eq!(kcfg.kap_ctrl[3], 9.5);
    assert_eq!(kcfg.kap_integer_ctrl[0], 7);
    assert!(kcfg.kap_logical_ctrl[1]);
    assert_eq!(kcfg.kap_character_ctrl[2], "op");
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);
}

#[test]
fn test_kap_namelist_rejects_unbundled_tables_hooks_and_invalid_type2_bounds() {
    let tmp = TempCaseGuard::new("kap_loud");
    for (tag, body, needle) in [
        (
            "aesopus_missing",
            "&kap\n  kap_lowT_prefix = 'AESOPUS'\n/\n",
            "aesopus_filename",
        ),
        (
            "aesopus_nonexistent",
            "&kap\n  kap_lowT_prefix = 'AESOPUS'\n  AESOPUS_filename = 'AESOPUS_NONEXISTENT_TEST.h5'\n/\n",
            "AESOPUS_NONEXISTENT_TEST.h5",
        ),
        (
            "custom_prefix",
            "&kap\n  kap_file_prefix = 'custom_user_table'\n/\n",
            "kap_file_prefix",
        ),
        (
            "hook_rad",
            "&kap\n  use_other_radiative_opacity = .true.\n/\n",
            "use_other_radiative_opacity",
        ),
        (
            "hook_cond",
            "&kap\n  use_other_elect_cond_opacity = .true.\n/\n",
            "use_other_elect_cond_opacity",
        ),
        (
            "hook_comp",
            "&kap\n  use_other_compton_opacity = .true.\n/\n",
            "use_other_compton_opacity",
        ),
        (
            "type2_gt_071",
            "&kap\n  kap_Type2_full_off_X = 0.75d0\n/\n",
            "0.71",
        ),
        (
            "type2_x_order",
            "&kap\n  kap_Type2_full_off_X = 0.30d0\n  kap_Type2_full_on_X = 0.50d0\n/\n",
            "kap_type2_full_off_x",
        ),
        (
            "type2_dz_order",
            "&kap\n  kap_Type2_full_off_dZ = 0.02d0\n  kap_Type2_full_on_dZ = 0.005d0\n/\n",
            "kap_type2_full_off_dz",
        ),
    ] {
        let err = tmp
            .load_inlist(&format!("inlist_{tag}"), body)
            .expect_err(tag);
        assert!(
            matches!(err, StarError::InvalidInput(_)),
            "{tag}: unexpected variant {err:?}"
        );
        assert!(
            err.to_string().contains(needle),
            "{tag}: expected '{needle}' in '{err}'"
        );
    }

    // Missing AESOPUS_filename must STILL fail loudly even with allow_unsupported_controls = true
    let bad_path = tmp.root.join("inlist_aesopus_allow_unsupported");
    fs::write(
        &bad_path,
        "&kap\n  kap_lowT_prefix = 'AESOPUS'\n  AESOPUS_filename = 'AESOPUS_NONEXISTENT_TEST.h5'\n/\n",
    )
    .expect("write inlist");
    let err = InlistBundle::from_file_in_case_dir_with_options(
        &tmp.root,
        "inlist_aesopus_allow_unsupported",
        flexmesa_star::inlist::InlistLoadOptions {
            allow_unsupported_controls: true,
        },
    )
    .expect_err("missing AESOPUS_filename must not silently downgrade");
    assert!(err.to_string().contains("AESOPUS_NONEXISTENT_TEST.h5"));
}

#[test]
fn test_star_job_filename_of_special_rate_and_custom_aesopus_inlist() {
    let tmp = TempCaseGuard::new("rcrb_special_rate_aesopus");

    // 1. Built-in special rate tables resolve cleanly and wire into resolve_model_net
    let bundle = tmp
        .load_inlist(
            "inlist_special",
            r"
&star_job
  num_special_rate_factors = 2
  reaction_for_special_factor(1) = 'r_c12_ag_o16'
  special_rate_factor(1) = 1.0d0
  filename_of_special_rate(1) = 'c12ag_deboer_sigma_0p0_2000_Tgrid.dat'
  reaction_for_special_factor(2) = 'r_he4_he4_he4_to_c12'
  special_rate_factor(2) = 1.0d0
  filename_of_special_rate(2) = 'r_he4_he4_he4_to_c12_cf88.txt'
/
",
        )
        .expect("parse filename_of_special_rate");
    assert_eq!(bundle.job.num_special_rate_factors, 2);
    assert_eq!(
        bundle.job.filename_of_special_rate,
        vec![
            "c12ag_deboer_sigma_0p0_2000_Tgrid.dat",
            "r_he4_he4_he4_to_c12_cf88.txt",
        ]
    );
    assert_eq!(bundle.validation_report.unsupported_controls.len(), 0);

    // 2. Case-dir local special rate file resolves to case_dir path
    fs::write(
        tmp.root.join("local_rate.dat"),
        "# custom local rate\n3\n0.1 1.0e-12\n1.0 2.0e-10\n10.0 5.0e-8\n",
    )
    .expect("write local rate");
    let local_bundle = tmp
        .load_inlist(
            "inlist_local_rate",
            "&star_job\n  num_special_rate_factors = 1\n  reaction_for_special_factor(1) = 'r_c12_ag_o16'\n  filename_of_special_rate(1) = 'local_rate.dat'\n/\n",
        )
        .expect("load local rate inlist");
    assert!(local_bundle.job.filename_of_special_rate[0]
        .ends_with("local_rate.dat"));

    // 3. Missing special rate file fails loudly
    let err = tmp
        .load_inlist(
            "inlist_missing_rate",
            "&star_job\n  num_special_rate_factors = 1\n  reaction_for_special_factor(1) = 'r_c12_ag_o16'\n  filename_of_special_rate(1) = 'missing_special_rate_xyz.dat'\n/\n",
        )
        .expect_err("missing filename_of_special_rate must fail loudly");
    assert!(err.to_string().contains("missing_special_rate_xyz.dat"));

    // 4. Custom AESOPUS HDF5 file in case_dir stays KapLowTOption::Aesopus
    if let Some(rcrb_dir) = flexmesa_utils::require_mesa_test_input_path(
        "star/test_suite/R_CrB_star",
    ) {
        let rcrb_bundle = InlistBundle::from_file_in_case_dir_with_options(
            &rcrb_dir,
            "inlist_R_CrB_star_header",
            flexmesa_star::inlist::InlistLoadOptions {
                allow_unsupported_controls: true,
            },
        )
        .expect("load R_CrB_star inlist");
        assert_eq!(
            rcrb_bundle.kap_config.kap_low_t_option,
            KapLowTOption::Aesopus
        );
        assert!(rcrb_bundle
            .kap_config
            .aesopus_filename
            .ends_with("AESOPUS_GS98_RCrB.h5"));
        assert_eq!(
            rcrb_bundle.job.filename_of_special_rate,
            vec![
                "c12ag_deboer_sigma_0p0_2000_Tgrid.dat",
                "r_he4_he4_he4_to_c12_cf88.txt",
            ]
        );
    }
}

#[test]
fn test_filename_of_special_rate_wires_resolve_model_net_and_invalidates_micro_cache() {
    let mut model = flexmesa_star::zams::load_zams_model(
        flexmesa_star::StarControls::default(),
        flexmesa_star::StarJobControls::default(),
    )
    .expect("load zams");
    // Heat core into He-burning regime so c12(a,g)o16 and 3a are active
    model.kap_config.zbase = 0.02;
    model.controls.zbase = 0.02;
    let nz = model.nz;
    let sp = model.species;
    let ihe4 = model.iso_names.iter().position(|n| n == "he4").unwrap();
    let ic12 = model.iso_names.iter().position(|n| n == "c12").unwrap();
    for k in (nz - 5)..nz {
        model.ln_t[k] = 18.890_684_555_543_35; // ln(1.6e8 K)
        model.lnd[k] = 9.210_340_371_976_184; // ln(1.0e4 g cm^-3)
        for j in 0..sp {
            model.xa[k * sp + j] = 0.0;
        }
        model.xa[k * sp + ihe4] = 0.5;
        model.xa[k * sp + ic12] = 0.5;
    }

    let net_base = flexmesa_star::micro::resolve_model_net(&model)
        .expect("resolve base net");
    flexmesa_star::hydro_vars::set_hydro_vars(&mut model, false, false)
        .expect("set_hydro_vars base");
    let eps_base = model.eps_nuc[nz - 1];

    model.job.num_special_rate_factors = 2;
    model.job.reaction_for_special_factor = vec![
        String::from("r_c12_ag_o16"),
        String::from("r_he4_he4_he4_to_c12"),
    ];
    model.job.special_rate_factor = vec![1.0, 1.0];
    model.job.filename_of_special_rate = vec![
        String::from("c12ag_deboer_sigma_0p0_2000_Tgrid.dat"),
        String::from("r_he4_he4_he4_to_c12_cf88.txt"),
    ];

    let net_override = flexmesa_star::micro::resolve_model_net(&model)
        .expect("resolve overridden net");
    assert_ne!(net_base.identity_id, net_override.identity_id);

    // Must invalidate MicrophysicsCache and recompute eps_nuc with overridden tables
    flexmesa_star::hydro_vars::set_hydro_vars(&mut model, false, false)
        .expect("set_hydro_vars override");
    let eps_override = model.eps_nuc[nz - 1];
    assert!(
        eps_base > 0.0 && eps_override > 0.0,
        "eps_base={eps_base:e} eps_override={eps_override:e}"
    );
    assert!(
        ((eps_override - eps_base) / eps_base).abs() > 1.0e-3,
        "MicrophysicsCache failed to invalidate or override rate table: base={eps_base:e} ov={eps_override:e}"
    );
}
