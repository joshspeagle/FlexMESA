// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration and round-trip tests for MESA `26.4.1` `LOGS/history.data`,
//! `LOGS/profileN.data`, and `LOGS/profiles.index` writers and readers (`#163`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct TempTestDir(PathBuf);

impl TempTestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "flexmesa_star_163_{label}_{}_{n}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

use flexmesa_chem::category_id;
use flexmesa_const::{MixingType, L_SUN, SECYER};
use flexmesa_star::{
    default_star_history_specs, evaluate_star_history_row, extract_history_row,
    format_mesa_dbl, format_mesa_int, load_zams_model,
    parse_star_history_columns, parse_star_profile_columns, read_mesa_log_file,
    read_mesa_log_str, read_profiles_index_file, read_profiles_index_str,
    safe_log10, set_hydro_vars, star_evolve_step, IncrementalHistoryWriter,
    MesaCellValue, MesaHeaderValue, OutputControls, StarControls, StarError,
    StarJobControls, StarLogsWriter,
};

/// Verbatim 10-line output produced by upstream Fortran MESA `26.4.1` (`gfortran 15.2.0`,
/// `MESA SDK x86_64-linux-26.6.1`, `CRMATH`) on `star/test` (`LOGS/history.data`).
const REAL_MESA_26_4_1_STAR_TEST_HISTORY: &str = "\
                                       1                                        2                                        3                                        4                                        5                                        6                                        7                                        8                                        9                                       10                                       11 
                          version_number                                 compiler                                    build                         MESA_SDK_version                             math_backend                                     date                                burn_min1                                burn_min2                                     msun                                     rsun                                     lsun 
                               \"r26.4.1\"                               \"gfortran\"                                 \"15.2.0\"                    \"x86_64-linux-26.6.1\"                                 \"CRMATH\"                               \"20261115\"                  5.0000000000000000E+001                  1.0000000000000000E+003                  1.9884098706980504E+033                  6.9570000000000000E+010                  3.8280000000000003E+033 

                                       1                                        2                                        3                                        4                                        5                                        6                                        7                                        8                                        9                                       10                                       11                                       12                                       13                                       14                                       15                                       16                                       17                                       18                                       19                                       20                                       21                                       22                                       23                                       24                                       25                                       26                                       27                                       28                                       29                                       30                                       31                                       32                                       33                                       34                                       35                                       36                                       37                                       38                                       39                                       40                                       41                                       42                                       43                                       44                                       45                                       46                                       47                                       48                                       49                                       50                                       51                                       52                                       53                                       54                                       55                                       56                                       57                                       58                                       59                                       60 
                            model_number                                num_zones                                 star_age                                   log_dt                                star_mass                               log_xmstar                             log_abs_mdot                           mass_conv_core                             conv_mx1_top                             conv_mx1_bot                             conv_mx2_top                             conv_mx2_bot                                  mx1_top                                  mx1_bot                                  mx2_top                                  mx2_bot                       log_power_nuc_burn                           power_nuc_burn                                   log_LH                                  log_LHe                                   log_LZ                                 log_Lnuc                                     Lnuc                                       pp                                      cno                                tri_alpha                               epsnuc_M_1                               epsnuc_M_2                               epsnuc_M_3                               epsnuc_M_4                               epsnuc_M_5                               epsnuc_M_6                               epsnuc_M_7                               epsnuc_M_8                             he_core_mass                             co_core_mass                            one_core_mass                             fe_core_mass                   neutron_rich_core_mass                                 log_Teff                                    log_L                                    log_R                                    log_g                        v_div_csound_surf                               log_cntr_P                             log_cntr_Rho                               log_cntr_T                                center_mu                                center_ye                              center_abar                                center_h1                               center_he4                               center_c12                               center_o16                              surface_c12                              surface_o16                            total_mass_h1                           total_mass_he4                              num_retries                                num_iters 
                                       1                                     2464                  3.1250000000000005E+003                  3.4948500216800937E+000                  4.0000000000000000E+000                  3.3900565901688630E+001                 -9.8999999999999986E+001                  8.8327623010735568E-001                  2.2073838924941963E-001                  7.8127322307788849E-008                  9.9999999982745025E-001                  9.9999999947312157E-001                  2.2073838924941963E-001                  7.8127322307788849E-008                  9.9999999982745025E-001                  9.9999999947312157E-001                  2.3746817314331112E+000                  2.3696365036935427E+002                  2.3746817314331112E+000                 -3.4525621634463619E+001                 -9.8999999999999986E+001                  2.3746817314331112E+000                  2.3696365036935427E+002                  1.2539862454307353E+000                  2.3404775929014501E+000                 -3.4525621634463619E+001                  0.0000000000000000E+000                  0.0000000000000000E+000                  2.8504313200105948E+032                  1.3244710613248378E+033                 -2.0000000000000000E+001                 -2.0000000000000000E+001                 -2.0000000000000000E+001                 -2.0000000000000000E+001                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  4.1698363731360653E+000                  2.3744642077011577E+000                  3.7021251844294467E-001                  4.2997025817452057E+000                  0.0000000000000000E+000                  1.6983158868200547E+001                  1.4370590954340996E+000                  7.4124145557482217E+000                  6.1830803755850239E-001                  8.4849256460888067E-001                  1.3006873417765605E+000                  6.9698465072387150E-001                  2.8247727106593346E-001                  4.1410753289359172E-005                  9.1330393301320475E-003                  3.4468833607860759E-003                  9.3805276451495428E-003                  2.7968035031409166E+000                  1.1224004028953543E+000                                        0                                        3 
                                      11                                      992                  1.0000000000000000E+005                  4.2759817101445359E+000                  4.0000000000000000E+000                  3.3900565901688630E+001                 -9.8999999999999986E+001                  8.7165174350008434E-001                  2.1743938464920787E-001                  7.8127320031831649E-008                  9.9999999982745025E-001                  9.9999999947891793E-001                  2.1743938464920787E-001                  7.8127320031831649E-008                  9.9999999982745025E-001                  9.9999999947891793E-001                  2.3751527377308568E+000                  2.3722078445983041E+002                  2.3751527377308568E+000                 -3.4522428241910902E+001                 -9.8999999999999986E+001                  2.3751527377308568E+000                  2.3722078445983041E+002                  1.2508049803411379E+000                  2.3412465179715189E+000                 -3.4522428241910902E+001                  0.0000000000000000E+000                  0.0000000000000000E+000                  2.9337608936285420E+032                  1.3363764443602442E+033                 -2.0000000000000000E+001                 -2.0000000000000000E+001                 -2.0000000000000000E+001                 -2.0000000000000000E+001                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  4.1698543093610319E+000                  2.3748530992639290E+000                  3.7037109177439709E-001                  4.2993854350823009E+000                  0.0000000000000000E+000                  1.6983530260439228E+001                  1.4374477694970942E+000                  7.4124833425502450E+000                  6.1842925291720463E-001                  8.4836544208327125E-001                  1.3010086577706363E+000                  6.9673042024974685E-001                  2.8273473123506193E-001                  4.1562882659198952E-005                  9.1077544595079343E-003                  3.4468833607860759E-003                  9.3805276451495428E-003                  2.7965682165662291E+000                  1.1226285581611799E+000                                        0                                        3 
";

/// Verbatim header + first 3 outer zones from upstream Fortran MESA `26.4.1`
/// `star/test/LOGS/profile1.data`.
const REAL_MESA_26_4_1_STAR_TEST_PROFILE_EXCERPT: &str = "\
                                       1                                        2                                        3                                        4                                        5                                        6                                        7                                        8                                        9                                       10                                       11                                       12                                       13                                       14                                       15                                       16                                       17                                       18                                       19                                       20                                       21                                       22                                       23                                       24                                       25                                       26                                       27                                       28                                       29                                       30                                       31                                       32                                       33                                       34                                       35                                       36                                       37                                       38                                       39                                       40                                       41                                       42                                       43                                       44                                       45                                       46                                       47                                       48                                       49 
                            model_number                                num_zones                             initial_mass                                initial_z                                 star_age                                time_step                                     Teff                            photosphere_L                            photosphere_r                               center_eta                                center_h1                               center_he3                               center_he4                               center_c12                               center_n14                               center_o16                              center_ne20                                star_mass                                star_mdot                             star_mass_h1                            star_mass_he3                            star_mass_he4                            star_mass_c12                            star_mass_n14                            star_mass_o16                           star_mass_ne20                             he_core_mass                             co_core_mass                             fe_core_mass                   neutron_rich_core_mass                             dynamic_time                             kh_timescale                            nuc_timescale                           power_nuc_burn                             power_h_burn                            power_he_burn                                power_neu                                burn_min1                                burn_min2                             time_seconds                           version_number                                 compiler                                    build                         MESA_SDK_version                             math_backend                                     date                                     msun                                     rsun                                     lsun 
                                       1                                     2464                  4.0000000000000000E+000                  2.0000000000000000E-002                  3.1250000000000005E+003                  3.1250000000000005E+003                  1.4785512171929237E+004                  2.3684499283806815E+002                  2.3453762251799066E+000                 -3.8159739983414753E+000                  6.9698465072387150E-001                  1.4354816690010983E-006                  2.8247727106593346E-001                  4.1410753289359172E-005                  5.1999463919718183E-003                  9.1330393301320475E-003                  2.0850244238757956E-003                  4.0000000000000000E+000                  0.0000000000000000E+000                  2.7968035031409242E+000                  8.7709596062066753E-005                  1.1224004028953527E+000                  9.3531182972628503E-003                  9.4296490636994953E-003                  3.7276631994176378E-002                  8.3400976955009653E-003                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  0.0000000000000000E+000                  1.7974045568066078E+004                  6.7830366670284548E+005                  1.6888682982353845E+008                  2.3696365036935427E+002                  2.3696365036935430E+002                  2.9811124997774564E-035                  1.7118554685743199E+001                  5.0000000000000000E+001                  1.0000000000000000E+003                  9.8617500000000015E+010                                \"r26.4.1\"                               \"gfortran\"                                 \"15.2.0\"                    \"x86_64-linux-26.6.1\"                                 \"CRMATH\"                               \"20261115\"                  1.9884098706980504E+033                  6.9570000000000000E+010                  3.8280000000000003E+033 

                                       1                                        2                                        3                                        4                                        5                                        6                                        7                                        8                                        9                                       10                                       11                                       12 
                                    zone                                     mass                                     logR                                     logT                                   logRho                                     logP                        x_mass_fraction_H                       y_mass_fraction_He                   z_mass_fraction_metals                                       pp                                      cno                                tri_alpha 
                                       1                  4.0000000000000000E+000                  3.7021251844294467E-001                  4.1729181762195129E+000                 -9.0401365116830554E+000                  3.2514112412172045E+000                  6.9999999999999518E-001                  2.8000000000000502E-001                  1.9999999999999796E-002                  1.3369537640597871E-028                  0.0000000000000000E+000                  0.0000000000000000E+000 
                                       2                  3.9999999999983995E+000                  3.7018475489712405E-001                  4.1774608027585991E+000                 -9.0107464785605487E+000                  3.2847255883466029E+000                  6.9999999999999518E-001                  2.8000000000000502E-001                  1.9999999999999796E-002                  1.4305640072583255E-028                  0.0000000000000000E+000                  0.0000000000000000E+000 
                                       3                  3.9999999999976001E+000                  3.7017177963885306E-001                  4.1804337769147795E+000                 -8.9925576876133970E+000                  3.3055975596563307E+000                  6.9999999999999518E-001                  2.8000000000000502E-001                  1.9999999999999796E-002                  1.4917520387164350E-028                  0.0000000000000000E+000                  0.0000000000000000E+000 
";

/// Verbatim `star/test/LOGS/profiles.index` produced by upstream Fortran MESA `26.4.1`.
const REAL_MESA_26_4_1_STAR_TEST_PROFILES_INDEX: &str = concat!(
    "           2 models.    lines hold model number, priority, and profile number.\n",
    "           1           2           1\n",
    "          11           3           2\n",
);

#[test]
fn test_real_mesa_26_4_1_logs_roundtrip_and_default_column_alignment() {
    let hist = read_mesa_log_str(REAL_MESA_26_4_1_STAR_TEST_HISTORY)
        .expect("parse real MESA 26.4.1 history.data");
    assert_eq!(hist.header.len(), 11);
    assert_eq!(
        hist.header_value("version_number"),
        Some(&MesaHeaderValue::Text(String::from("r26.4.1")))
    );
    assert_eq!(hist.num_rows, 2);
    assert_eq!(hist.column_names.len(), 60);

    let default_specs = default_star_history_specs();
    let expected_names: Vec<String> = default_specs
        .iter()
        .map(flexmesa_star::HistoryColumnSpec::column_name)
        .collect();
    assert_eq!(hist.column_names, expected_names);

    let orig_row1 = REAL_MESA_26_4_1_STAR_TEST_HISTORY
        .lines()
        .nth(6)
        .expect("row 1");
    let int_cols = ["model_number", "num_zones", "num_retries", "num_iters"];
    let mut reformatted_row1 = String::new();
    for (j, cname) in hist.column_names.iter().enumerate() {
        let v = hist.column_values[j][0];
        if int_cols.contains(&cname.as_str()) {
            reformatted_row1.push_str(&format_mesa_int(v as i64));
        } else {
            reformatted_row1.push_str(&format_mesa_dbl(v));
        }
    }
    assert_eq!(reformatted_row1, orig_row1);

    let prof = read_mesa_log_str(REAL_MESA_26_4_1_STAR_TEST_PROFILE_EXCERPT)
        .expect("parse real MESA 26.4.1 profile1.data");
    assert_eq!(prof.header.len(), 49);
    assert_eq!(prof.column_names.len(), 12);
    assert_eq!(prof.num_rows, 3);
    assert_eq!(
        prof.header_value("model_number"),
        Some(&MesaHeaderValue::Int(1))
    );
    assert_eq!(
        prof.header_value("num_zones"),
        Some(&MesaHeaderValue::Int(2464))
    );

    let idx =
        read_profiles_index_str(REAL_MESA_26_4_1_STAR_TEST_PROFILES_INDEX)
            .expect("parse real MESA 26.4.1 profiles.index");
    assert_eq!(idx.entries.len(), 2);
    assert_eq!(idx.format_text(), REAL_MESA_26_4_1_STAR_TEST_PROFILES_INDEX);
}

#[test]
fn test_f4_28_log_lh_separates_hydrogen_helium_and_metal_burning_powers() {
    let controls = StarControls {
        initial_mass: 2.0,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    model.dt = 100.0 * SECYER;
    set_hydro_vars(&mut model, false, false).expect("hydro");

    let i_pp = category_id("pp") - 1;
    let i_cno = category_id("cno") - 1;
    let i_3a = category_id("tri_alpha") - 1;
    let i_ca = category_id("c_alpha") - 1;
    let i_pn = category_id("photo") - 1;

    let m_tot = model.dm.iter().sum::<f64>();
    let target_lh_lsun = 12.5_f64;
    let target_lhe_lsun = 85.0_f64;
    let target_lz_lsun = 0.4_f64;
    let target_photo_lsun = -0.1_f64;

    for k in 0..model.nz {
        model.aux.eps_nuc_categories[k].fill(0.0);
        model.aux.eps_nuc_categories[k][i_pp] =
            0.25 * target_lh_lsun * L_SUN / m_tot;
        model.aux.eps_nuc_categories[k][i_cno] =
            0.75 * target_lh_lsun * L_SUN / m_tot;
        model.aux.eps_nuc_categories[k][i_3a] = target_lhe_lsun * L_SUN / m_tot;
        model.aux.eps_nuc_categories[k][i_ca] = target_lz_lsun * L_SUN / m_tot;
        model.aux.eps_nuc_categories[k][i_pn] =
            target_photo_lsun * L_SUN / m_tot;
        model.eps_nuc[k] = model.aux.eps_nuc_categories[k].iter().sum();
    }

    // Configure an inner convective core (`k >= nz/2`) so `mass_conv_core > 0`.
    let k_core = model.nz / 2;
    for k in 0..model.nz {
        if k >= k_core {
            model.mixing_type[k] = MixingType::ConvectiveMixing;
            model.d_mix[k] = 1.0e14;
            model.gradr[k] = 0.45;
            model.gradl[k] = 0.40;
        } else {
            model.mixing_type[k] = MixingType::NoMixing;
            model.d_mix[k] = 0.0;
            model.gradr[k] = 0.25;
            model.gradl[k] = 0.40;
        }
    }

    let hr = extract_history_row(&model);
    assert!(
        (hr.log_lh - safe_log10(target_lh_lsun)).abs() < 1.0e-12,
        "F4-28 regression: log_lh={} != log10(L_H)={}",
        hr.log_lh,
        safe_log10(target_lh_lsun)
    );
    assert!((hr.log_lhe - safe_log10(target_lhe_lsun)).abs() < 1.0e-12);
    // `report.f90:101`: `power_z_burn = power_nuc_burn - (power_h_burn + power_he_burn)`,
    // while `history.f90:1615`: `Lnuc = power_nuc_burn - power_photo`.
    let expected_lz_lsun = target_lz_lsun + target_photo_lsun;
    assert!((hr.log_lz - safe_log10(expected_lz_lsun)).abs() < 1.0e-12);
    let expected_lnuc_nonphoto =
        target_lh_lsun + target_lhe_lsun + target_lz_lsun;
    assert!((hr.log_lnuc - safe_log10(expected_lnuc_nonphoto)).abs() < 1.0e-12);
    assert!(hr.mass_conv_core > 0.0 && hr.mass_conv_core < hr.star_mass);

    let custom_list = "\
        include ''\n\
        mixing_regions 3\n\
        burning_regions 2\n\
    ";
    let specs = parse_star_history_columns(custom_list, None, &model)
        .expect("parse custom history columns");
    let row = evaluate_star_history_row(&model, &specs);
    let find_f64 = |col: &str| -> f64 {
        row.iter()
            .find_map(|(n, v)| {
                (n == col).then_some(match *v {
                    MesaCellValue::Float(x) => x,
                    MesaCellValue::Int(i) => i as f64,
                })
            })
            .unwrap_or_else(|| panic!("missing column {col}"))
    };
    assert!((find_f64("log_LH") - safe_log10(target_lh_lsun)).abs() < 1.0e-12);
    assert!(
        (find_f64("log_LHe") - safe_log10(target_lhe_lsun)).abs() < 1.0e-12
    );
    assert!(
        (find_f64("log_LZ") - safe_log10(expected_lz_lsun)).abs() < 1.0e-12
    );
    assert!(
        (find_f64("log_Lnuc") - safe_log10(expected_lnuc_nonphoto)).abs()
            < 1.0e-12
    );
    assert!(
        (find_f64("tri_alpha") - safe_log10(target_lhe_lsun)).abs() < 1.0e-12
    );
    assert_eq!(find_f64("mix_type_1") as i64, 1);
    assert!(find_f64("mix_qtop_1") > 0.0 && find_f64("mix_qtop_1") < 1.0);
}

#[test]
fn test_incremental_flush_survives_mid_run_drop_and_profiles_evict_by_priority()
{
    let tmp = TempTestDir::new("flush_evict");
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        output: OutputControls {
            history_interval: 3,
            profile_interval: 2,
            priority_profile_interval: 4,
            max_num_profile_models: 2,
            ..OutputControls::default()
        },
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    model.dt = 1.0e3 * SECYER;
    set_hydro_vars(&mut model, false, false).expect("hydro");
    model.save_start_values();

    let abort_hist_path = tmp.path().join("aborted_history.data");
    {
        let mut writer = IncrementalHistoryWriter::new(
            abort_hist_path.clone(),
            default_star_history_specs(),
        );
        model.model_number = 1;
        writer.write_step(&model).expect("step 1");
        model.model_number = 2;
        model.star_age += 100.0;
        writer.write_step(&model).expect("step 2");
        // Drop `writer` without calling any explicit close method.
    }
    let recovered =
        read_mesa_log_file(&abort_hist_path).expect("read flushed partial log");
    assert_eq!(recovered.num_rows, 2);
    assert_eq!(recovered.column("model_number"), Some(&[1.0, 2.0][..]));

    // Exercise `StarLogsWriter` priority eviction across steps 1..=5:
    // - step 1 (`is_first_step = true`): priority-2 profile1 + history
    // - step 2 (`% 2 == 0`): priority-1 profile2 + history (because `prof_due`)
    // - step 3 (`% 3 == 0`): history only
    // - step 4 (`% 4 == 0`): priority-2 profile -> evicts priority-1 model 2 (reusing slot 2!)
    // - step 5 (`is_terminating = true`): priority-3 profile -> evicts oldest priority-2 model 1 (reusing slot 1!)
    let mut logs =
        StarLogsWriter::for_stage(&model, tmp.path(), tmp.path(), None, None)
            .expect("init StarLogsWriter");
    for step in 1..=5_usize {
        model.model_number = step;
        model.star_age = (step as f64) * 1.0e3;
        logs.record_step(&model, step == 1, step == 5)
            .expect("record step");
    }

    let log_dir = tmp.path().join("LOGS");
    let hist_tbl = read_mesa_log_file(&log_dir.join("history.data"))
        .expect("history.data");
    assert_eq!(
        hist_tbl.column("model_number"),
        Some(&[1.0, 2.0, 3.0, 4.0, 5.0][..])
    );

    let idx = read_profiles_index_file(&log_dir.join("profiles.index"))
        .expect("profiles.index");
    assert_eq!(idx.entries.len(), 2);
    assert_eq!(idx.entries[0].model_number, 4);
    assert_eq!(idx.entries[0].priority, 2);
    assert_eq!(idx.entries[0].profile_number, 2);
    assert_eq!(idx.entries[1].model_number, 5);
    assert_eq!(idx.entries[1].priority, 3);
    assert_eq!(idx.entries[1].profile_number, 1);

    let p1 =
        read_mesa_log_file(&log_dir.join("profile1.data")).expect("profile1");
    assert_eq!(
        p1.header_value("model_number"),
        Some(&MesaHeaderValue::Int(5))
    );
    assert_eq!(p1.num_rows, model.nz);
}

#[test]
fn test_unknown_history_and_profile_columns_fail_loudly() {
    let model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    let err_h = parse_star_history_columns(
        "model_number\nnot_a_mesa_col\n",
        None,
        &model,
    )
    .expect_err("unknown history column must fail");
    assert!(matches!(err_h, StarError::InvalidInput(_)));

    let err_p =
        parse_star_profile_columns("zone\nnot_a_mesa_profile_col\n", None)
            .expect_err("unknown profile column must fail");
    assert!(matches!(err_p, StarError::InvalidInput(_)));
}

#[test]
fn test_two_step_evolve_writes_mesa_logs_and_split_headers() {
    let tmp = TempTestDir::new("two_step");
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        output: OutputControls {
            history_interval: 1,
            profile_interval: 2,
            ..OutputControls::default()
        },
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(controls, StarJobControls::default()).expect("zams");
    model.dt = 1.0e4 * SECYER;
    model.dt_next = model.dt;
    set_hydro_vars(&mut model, false, false).expect("hydro");
    model.save_start_values();

    let mut logs =
        StarLogsWriter::for_stage(&model, tmp.path(), tmp.path(), None, None)
            .expect("logs");
    for step in 1..=2_usize {
        star_evolve_step(&mut model, false, true, true).expect("evolve step");
        model.mstar_start = model.mstar;
        model.sync_mesh_and_pack().expect("sync");
        set_hydro_vars(&mut model, false, false).expect("hydro");
        model.save_start_values();
        logs.record_step(&model, step == 1, step == 2)
            .expect("record");
    }

    let hist =
        read_mesa_log_file(&tmp.path().join("LOGS").join("history.data"))
            .expect("read written history.data");
    assert_eq!(hist.num_rows, 2);
    assert!(hist.column("star_age").expect("star_age")[1] > 0.0);

    // Verify split-header support (`star_history_header_name` & `profile_data_header_suffix`)
    // and stale-file clearing across stages.
    let split_tmp = TempTestDir::new("split_headers");
    let mut split_model = model;
    split_model.controls.output.star_history_header_name =
        String::from("history.hdr");
    split_model.controls.output.profile_data_header_suffix =
        String::from(".hdr");
    split_model.controls.output.write_profiles_flag = true;
    split_model.controls.output.profile_interval = 1;
    let mut split_logs = StarLogsWriter::for_stage(
        &split_model,
        split_tmp.path(),
        split_tmp.path(),
        Some(1),
        Some(true),
    )
    .expect("split logs");
    split_logs
        .record_step(&split_model, true, true)
        .expect("record split");
    let log_dir = split_tmp.path().join("LOGS");
    let h_hdr =
        fs::read_to_string(log_dir.join("history.hdr")).expect("history.hdr");
    let h_dat =
        fs::read_to_string(log_dir.join("history.data")).expect("history.data");
    assert_eq!(h_hdr.lines().count(), 6);
    assert_eq!(h_dat.lines().count(), 1);
    let combined_h = format!("{h_hdr}{h_dat}");
    assert_eq!(
        read_mesa_log_str(&combined_h)
            .expect("parse split hist")
            .num_rows,
        1
    );

    let p_hdr = fs::read_to_string(log_dir.join("profile1.data.hdr"))
        .expect("profile1.data.hdr");
    let p_dat = fs::read_to_string(log_dir.join("profile1.data"))
        .expect("profile1.data");
    assert_eq!(p_hdr.lines().count(), 6);
    assert_eq!(p_dat.lines().count(), split_model.nz);

    let _fresh_logs = StarLogsWriter::for_stage(
        &split_model,
        split_tmp.path(),
        split_tmp.path(),
        Some(1),
        Some(false),
    )
    .expect("fresh logs");
    assert!(!log_dir.join("history.data").exists());
    assert!(!log_dir.join("history.hdr").exists());
}

#[test]
fn test_multistage_rn_logs_dir_routing() {
    let ms_case = TempTestDir::new("multistage_case");
    let ms_out = TempTestDir::new("multistage_out");
    fs::write(
        ms_case.path().join("rn"),
        "#!/bin/sh\ndo_one inlist_s1 s1.mod LOGS_s1\ndo_one inlist_s2 s2.mod LOGS_s2\n",
    )
    .expect("write rn");
    fs::write(
        ms_case.path().join("inlist_s1"),
        "&star_job\n  steps_to_take_before_terminate = 1\n/\n&controls\n  initial_mass = 1.0\n  initial_z = 0.02\n  history_interval = 1\n  write_profiles_flag = .false.\n/\n",
    )
    .expect("write inlist_s1");
    fs::write(
        ms_case.path().join("inlist_s2"),
        "&star_job\n  load_saved_model = .true.\n  saved_model_name = 's1.mod'\n  steps_to_take_before_terminate = 1\n/\n&controls\n  initial_mass = 1.0\n  initial_z = 0.02\n  history_interval = 1\n  write_profiles_flag = .false.\n/\n",
    )
    .expect("write inlist_s2");
    let rep = flexmesa_star::run_test_suite_case_with_logs(
        ms_case.path(),
        true,
        Some(1),
        false,
        flexmesa_star::StarLogsOptions {
            output_root: Some(ms_out.path()),
            explicit_history_file: None,
            history_interval_override: Some(1),
            write_profiles_override: Some(false),
        },
    )
    .expect("run 2-stage case");
    assert_eq!(rep.stages.len(), 2);
    let h1 =
        read_mesa_log_file(&ms_out.path().join("LOGS_s1").join("history.data"))
            .expect("LOGS_s1/history.data");
    let h2 =
        read_mesa_log_file(&ms_out.path().join("LOGS_s2").join("history.data"))
            .expect("LOGS_s2/history.data");
    assert_eq!(h1.num_rows, 1);
    assert_eq!(h2.num_rows, 1);
    assert!(!ms_out.path().join("LOGS").exists());
    let _ = fs::remove_file(ms_case.path().join("s1.mod"));
    let _ = fs::remove_file(ms_case.path().join("s2.mod"));
}

fn resolve_test_mesa_root() -> Option<PathBuf> {
    std::env::var("MESA_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.join("star/test_suite").is_dir())
        .or_else(|| {
            std::env::var("XDG_CACHE_HOME")
                .ok()
                .map(|p| PathBuf::from(p).join("flexmesa/mesa_test_inputs"))
                .filter(|p| p.join("star/test_suite").is_dir())
        })
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|p| {
                    PathBuf::from(p).join(".cache/flexmesa/mesa_test_inputs")
                })
                .filter(|p| p.join("star/test_suite").is_dir())
        })
}

fn case_history_column_files(case_dir: &std::path::Path) -> Vec<String> {
    let mut candidate_files = vec![
        String::from("history_columns.list"),
        String::from("history_columns_rsp.list"),
        String::from("astero_history_columns.list"),
    ];
    if let Ok(stages) = flexmesa_star::parse_rn_stages(case_dir, false) {
        for st in stages {
            if let flexmesa_star::RnStage::DoOne { inlist, .. } = st {
                if let Ok(b) =
                    flexmesa_star::InlistBundle::from_file_in_case_dir(
                        case_dir, &inlist,
                    )
                {
                    let hcf = b.job.history_columns_file.trim();
                    if !hcf.is_empty()
                        && !candidate_files.iter().any(|f| f == hcf)
                    {
                        candidate_files.push(hcf.to_owned());
                    }
                }
            }
        }
    }
    candidate_files
}

#[test]
fn test_upstream_test_suite_history_column_lists_parse_or_fail_loudly() {
    let model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    let mesa_root = resolve_test_mesa_root();
    assert!(
        mesa_root.is_some()
            || std::env::var("FLEXMESA_REQUIRE_ZERO_SKIPS").as_deref() != Ok("1"),
        "FLEXMESA_REQUIRE_ZERO_SKIPS=1 set, but neither MESA_DIR nor cached mesa_test_inputs was found"
    );
    let Some(base) = mesa_root else {
        return;
    };

    // Unported columns (#157 MLT++, #158 darkening/spots, #159 mass-add, #161 hydro/colors):
    let expected_unported = [
        "20M_z2m2_high_rotation",
        "adjust_net",
        "cburn_inward",
        "ccsn_IIp",
        "custom_colors",
        "double_bh",
        "gyre_in_mesa_wd",
        "high_rot_darkening",
        "magnetic_braking",
        "make_co_wd",
        "make_o_ne_wd",
        "ns_c",
        "pisn",
        "starspots",
        "tzo",
        "wd_acc_small_dm",
    ];
    let mut unexpected_failures = Vec::new();
    let mut loud_error_cases = std::collections::BTreeSet::new();
    for rel_suite in
        ["star/test_suite", "binary/test_suite", "astero/test_suite"]
    {
        let suite = base.join(rel_suite);
        if !suite.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&suite).expect("read suite") {
            let case_dir = entry.expect("entry").path();
            if !case_dir.is_dir() {
                continue;
            }
            let cname =
                case_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
            for fname in case_history_column_files(&case_dir) {
                let p = case_dir.join(&fname);
                if p.is_file() {
                    let txt = fs::read_to_string(&p).expect("read list");
                    match parse_star_history_columns(
                        &txt,
                        Some(&case_dir),
                        &model,
                    ) {
                        Ok(_) => {}
                        Err(StarError::InvalidInput(_))
                            if expected_unported.contains(&cname) =>
                        {
                            loud_error_cases.insert(cname.to_owned());
                        }
                        Err(e) => {
                            unexpected_failures
                                .push(format!("{}: {e:?}", p.display()));
                        }
                    }
                }
            }
        }
    }
    assert!(
        unexpected_failures.is_empty(),
        "unexpected history_columns.list parse failures:\n{}",
        unexpected_failures.join("\n")
    );
    if base.join("binary/test_suite/double_bh").is_dir() {
        let expected_present: std::collections::BTreeSet<String> =
            expected_unported
                .iter()
                .copied()
                .filter(|c| {
                    base.join("star/test_suite").join(c).is_dir()
                        || base.join("binary/test_suite").join(c).is_dir()
                })
                .map(str::to_owned)
                .collect();
        assert_eq!(loud_error_cases, expected_present);
    }
}

#[test]
fn test_numerics_mode_recorded_in_history_profile_mod_and_photo() {
    use flexmesa_math::{with_numerics_mode, NumericsMode};
    use flexmesa_star::{
        default_star_profile_specs, format_star_history_file_header,
        format_star_profile_text, photo_numerics_mode, write_mod_string,
        write_photo_bytes,
    };

    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");

    let hist_specs = default_star_history_specs();
    let prof_specs = default_star_profile_specs();

    for (mode, expected) in [
        (NumericsMode::Strict, "strict"),
        (NumericsMode::Fast, "fast"),
    ] {
        with_numerics_mode(mode, || {
            let h_hdr = format_star_history_file_header(&model, &hist_specs);
            let h_tbl = read_mesa_log_str(&format!(
                "{h_hdr}{}\n",
                " 1".repeat(hist_specs.len())
            ))
            .expect("parse history header");
            assert_eq!(
                h_tbl.header_value("math_backend"),
                Some(&MesaHeaderValue::Text(String::from(expected)))
            );

            let p_txt = format_star_profile_text(&model, &prof_specs);
            let p_tbl = read_mesa_log_str(&p_txt).expect("parse profile");
            assert_eq!(
                p_tbl.header_value("math_backend"),
                Some(&MesaHeaderValue::Text(String::from(expected)))
            );

            let mod_txt = write_mod_string(&model).expect("write_mod_string");
            assert!(
                mod_txt
                    .contains(&format!("! flexmesa math_backend: {expected}")),
                "missing math_backend comment in .mod for {expected}"
            );

            let photo_bytes = write_photo_bytes(&model);
            assert_eq!(
                photo_numerics_mode(&photo_bytes).expect("photo mode"),
                mode
            );
        });
    }
}
