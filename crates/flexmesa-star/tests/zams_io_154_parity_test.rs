// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone parity tests for ZAMS loading, pre-MS composition options,
//! `.mod`/`.photo` checkpoint serialization, [`PrimaryStateBuffer`] rollback
//! coverage, and `./rn` shell script parsing (issue #154).

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use flexmesa_const::SECYER;
use flexmesa_math::log;
use flexmesa_star::rotation_mix_info::StStartInfo;
use flexmesa_star::{
    build_pre_ms_model, compute_photo_checksum, interp_q, load_photo_bytes,
    load_zams_model, load_zams_model_from_catalog, parse_rn_stages,
    read_mod_str, read_photo_bytes, write_mod_string, write_photo_bytes,
    yrs_for_init_timestep, PrimaryStateBuffer, RnStage, StarControls,
    StarError, StarJobControls, ZamsCatalog, ZamsEntry,
};

static TEMP_SEQ: AtomicUsize = AtomicUsize::new(0);

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir()
            .join(format!("flexmesa_154_{tag}_{}_{seq}", std::process::id()));
        fs::create_dir_all(&path).expect("create temp test dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn reverse_entry_species(entry: &ZamsEntry) -> ZamsEntry {
    let mut rev = entry.clone();
    rev.chem_id.reverse();
    let sp = rev.species;
    for k in 0..rev.nz {
        for j in 0..sp {
            rev.xa[k * sp + j] = entry.xa[k * sp + (sp - 1 - j)];
        }
    }
    rev
}

fn minimal_two_zone_mod(
    file_type: u32,
    header_extra: &str,
    prev_block: &str,
) -> String {
    format!(
        "! synthetic mod fixture\n\
         {file_type:14} -- model for mesa/star\n\
         \n\
                      M/Msun      1.0000000000000000E+00\n\
                model_number                           5\n\
                    star_age      1.0000000000000000E+06\n\
                   initial_z      2.0000000000000000E-02\n\
                    n_shells                           2\n\
                    net_name   'basic.net'\n\
                     species                           8\n\
                        Teff      5.7770000000000000E+03\n\
         {header_extra}\
         \n\
                 lnd                       lnT                       lnR                       L                         dq                        h1                        he3                       he4                       c12                       n14                       o16                       ne20                      mg24\n\
             1   -1.0000000000000000E+01    1.0000000000000000E+01    2.4966216312141033E+01    3.8280000000000000E+33    5.0000000000000000E-01    7.0000000000000000E-01    1.0000000000000000E-05    2.7999000000000000E-01    4.0000000000000000E-03    2.0000000000000000E-03    8.0000000000000000E-03    2.0000000000000000E-03    4.0000000000000000E-03\n\
             2    2.0000000000000000E+00    1.6000000000000000E+01    2.4000000000000000E+01    1.9140000000000000E+33    5.0000000000000000E-01    7.0000000000000000E-01    1.0000000000000000E-05    2.7999000000000000E-01    4.0000000000000000E-03    2.0000000000000000E-03    8.0000000000000000E-03    2.0000000000000000E-03    4.0000000000000000E-03\n\
         {prev_block}"
    )
}

#[test]
fn zams_parse_ascii_stops_index_scan_at_any_non_positive_sentinel() {
    let model_a = ZamsCatalog::builtin()
        .models
        .first()
        .expect("builtin ZAMS non-empty");
    let sp_cols =
        ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "mg24"].join(" ");
    let mut text = format!(
        "net_name 'basic.net'\n\
         initial_z 0.02\n\
         initial_y 0.28\n\
         M/Msun n_shells\n\
         {:.8} {}\n\
         0.00000000 -1\n\
         999.0 5\n\
         \n",
        model_a.index_m_msun, model_a.nz
    );
    let _ = writeln!(text, "M/Msun {:.16E}", model_a.m_msun);
    let _ = writeln!(text, "n_shells {}", model_a.nz);
    let _ = writeln!(text, "lnd lnT lnR L dq {sp_cols}");
    for k in 0..model_a.nz {
        let _ = write!(
            text,
            "{} {:.16E} {:.16E} {:.16E} {:.16E} {:.16E}",
            k + 1,
            model_a.lnd[k],
            model_a.ln_t[k],
            model_a.ln_r[k],
            model_a.lum[k],
            model_a.dq[k]
        );
        for j in 0..model_a.species {
            let _ =
                write!(text, " {:.16E}", model_a.xa[k * model_a.species + j]);
        }
        text.push('\n');
    }

    let parsed = ZamsCatalog::parse_ascii(&text).expect("parse synthetic ZAMS");
    assert_eq!(parsed.models.len(), 1);
    assert_eq!(
        parsed.models[0].index_m_msun.to_bits(),
        model_a.index_m_msun.to_bits(),
        "stray 999.0 line after 0.00000000 sentinel must not be read"
    );
}

#[test]
fn zams_mass_interpolation_remaps_upper_species_by_chem_id_oracle() {
    let builtin = ZamsCatalog::builtin();
    let e1 = builtin
        .models
        .iter()
        .find(|m| (m.index_m_msun - 1.0).abs() < 0.05)
        .expect("near 1 Msun entry")
        .clone();
    let e2_orig = builtin
        .models
        .iter()
        .find(|m| m.index_m_msun > e1.index_m_msun + 0.15)
        .expect("higher mass entry")
        .clone();
    let e2_perm = reverse_entry_species(&e2_orig);

    let cat_orig = ZamsCatalog {
        initial_z: builtin.initial_z,
        initial_y: builtin.initial_y,
        net_name: builtin.net_name.clone(),
        models: vec![e1.clone(), e2_orig.clone()],
    };
    let cat_perm = ZamsCatalog {
        initial_z: builtin.initial_z,
        initial_y: builtin.initial_y,
        net_name: builtin.net_name.clone(),
        models: vec![e1.clone(), e2_perm],
    };

    let mid_mass = f64::midpoint(e1.index_m_msun, e2_orig.index_m_msun);
    let controls = StarControls {
        initial_mass: mid_mass,
        initial_z: 0.02,
        zbase: 0.02,
        ..StarControls::default()
    };
    let m_orig = load_zams_model_from_catalog(
        &cat_orig,
        controls.clone(),
        StarJobControls::default(),
    )
    .expect("orig catalog ZAMS");
    let m_perm = load_zams_model_from_catalog(
        &cat_perm,
        controls.clone(),
        StarJobControls::default(),
    )
    .expect("permuted upper bracket ZAMS");
    assert_eq!(m_perm.chem_id, m_orig.chem_id);
    for (idx, (&got, &want)) in m_perm.xa.iter().zip(&m_orig.xa).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "species mismatch at xa[{idx}] after chem_id permutation"
        );
    }

    // Closed-form log-mass interpolation check on outermost cell h1.
    let lnm1 = log(e1.index_m_msun);
    let lnm2 = log(e2_orig.index_m_msun);
    let alfa = (log(mid_mass) - lnm2) / (lnm1 - lnm2);
    let beta = 1.0 - alfa;
    let sp = e1.species;
    let x2_h1: Vec<f64> = (0..e2_orig.nz).map(|k| e2_orig.xa[k * sp]).collect();
    let x2_on_1 = interp_q(&e2_orig.dq, &x2_h1, &e1.dq).expect("interp_q");
    let expected_h1_0 = alfa * e1.xa[0] + beta * x2_on_1[0];
    assert_eq!(m_perm.xa[0].to_bits(), expected_h1_0.to_bits());

    // Missing species in upper bracket fails loudly instead of misindexing.
    let mut e2_bad = e2_orig;
    e2_bad.chem_id[sp - 1] += 1_000;
    let cat_bad = ZamsCatalog {
        initial_z: builtin.initial_z,
        initial_y: builtin.initial_y,
        net_name: builtin.net_name.clone(),
        models: vec![e1, e2_bad],
    };
    let err = load_zams_model_from_catalog(
        &cat_bad,
        controls,
        StarJobControls::default(),
    )
    .expect_err("missing chem_id in upper bracket must fail");
    assert!(matches!(err, StarError::InvalidInput(_)));
}

#[test]
fn zams_max_years_for_timestep_caps_dt_next_but_preserves_dt() {
    let dt_yrs = yrs_for_init_timestep(1.0);
    let cap_yrs = 0.01 * dt_yrs;
    let model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            zbase: 0.02,
            max_years_for_timestep: cap_yrs,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS with max_years_for_timestep");

    assert_eq!(model.dt.to_bits(), (dt_yrs * SECYER).to_bits());
    assert_eq!(model.dt_next.to_bits(), (cap_yrs * SECYER).to_bits());
}

#[test]
fn pre_ms_honors_job_dump_missing_metals_into_heaviest() {
    let controls = StarControls {
        initial_mass: 1.0,
        initial_z: 0.02,
        initial_y: 0.28,
        zbase: 0.02,
        ..StarControls::default()
    };
    let job_dump_true = StarJobControls {
        create_pre_ms_model: true,
        pre_ms_d_log10_p: 0.15,
        dump_missing_metals_into_heaviest: true,
        ..StarJobControls::default()
    };
    let job_dump_false = StarJobControls {
        dump_missing_metals_into_heaviest: false,
        ..job_dump_true.clone()
    };

    let m_dump = build_pre_ms_model(controls.clone(), job_dump_true)
        .expect("pre-MS with dump_missing_metals_into_heaviest = true");
    let m_prop = build_pre_ms_model(controls, job_dump_false)
        .expect("pre-MS with dump_missing_metals_into_heaviest = false");

    // `basic.net`: [h1(0), he3(1), he4(2), c12(3), n14(4), o16(5), ne20(6), mg24(7)].
    let xa_dump = m_dump.cell_xa(0);
    let xa_prop = m_prop.cell_xa(0);
    assert!(
        xa_dump[7] > xa_prop[7] + 1.0e-4,
        "heaviest metal mg24 must shrink when missing metals are not dumped into it: dump={}, prop={}",
        xa_dump[7],
        xa_prop[7]
    );
    assert!(
        xa_prop[5] > xa_dump[5] + 1.0e-4,
        "o16 must scale up when missing metals are distributed proportionally: prop={}, dump={}",
        xa_prop[5],
        xa_dump[5]
    );
    let metal_sum_dump: f64 = xa_dump[3..].iter().sum();
    let metal_sum_prop: f64 = xa_prop[3..].iter().sum();
    assert!((metal_sum_dump - 0.02).abs() <= 1.0e-12);
    assert!((metal_sum_prop - 0.02).abs() <= 1.0e-12);
}

#[test]
fn mod_io_roundtrips_dt_and_validates_lnpgas_and_previous_model() {
    // 1. Reject bit_for_lnPgas (file_type bit 1 => mask 2).
    let lnpgas_mod = minimal_two_zone_mod(
        2,
        "      year_for_time_step      1.0000000000000000E+03\n",
        "",
    );
    let err_lnpgas = read_mod_str(
        &lnpgas_mod,
        StarControls::default(),
        StarJobControls::default(),
    )
    .expect_err("lnPgas .mod must be rejected");
    assert!(matches!(err_lnpgas, StarError::IoError(_)));

    // 2. Unconditionally apply `timestep (seconds)` over `year_for_time_step`.
    let prev_override = "\n\
        previous model\n\
               previous n_shells                           2\n\
           previous mass (grams)      1.9884000000000000E+33\n\
              timestep (seconds)      4.3210000000000000E+09\n\
               dt_next (seconds)      6.5430000000000000E+09\n";
    let override_mod = minimal_two_zone_mod(
        4,
        "      year_for_time_step      1.0000000000000000E+03\n",
        prev_override,
    );
    let mut loaded = read_mod_str(
        &override_mod,
        StarControls::default(),
        StarJobControls::default(),
    )
    .expect("read .mod with previous model override");
    assert_eq!(loaded.dt.to_bits(), 4.321e9_f64.to_bits());
    assert_eq!(loaded.dt_next.to_bits(), 6.543e9_f64.to_bits());
    assert_eq!(
        loaded.aux.step_hist.time_sec.to_bits(),
        loaded.time_sec.to_bits()
    );

    // 3. A first-generation model has no previous model to write: upstream's
    // `prev_flag` (`write_model.f90:74-77`) clears `bit_for_2models` and the
    // block (`:218-231`), so the file reloads without the dt_next checks.
    assert_eq!(loaded.generations, 1);
    let first_gen = write_mod_string(&loaded).expect("write_mod_string");
    assert!(!first_gen.contains("previous model"));
    let header_ft: u32 = first_gen
        .lines()
        .find(|l| l.contains("-- model for mesa/star"))
        .and_then(|l| l.split_whitespace().next())
        .and_then(|t| t.parse().ok())
        .expect("file_type header");
    assert_eq!(header_ft & 4, 0);
    read_mod_str(&first_gen, loaded.controls.clone(), loaded.job.clone())
        .expect("first-generation .mod reloads");

    // 4. With a stored previous model (`generations > 1`, `nz_old == nz`),
    // write_mod_string emits `s% dt`, not `s% dt_old`, on `timestep (seconds)`.
    loaded.generations = 2;
    loaded.dt = 9.876_543_21e8;
    loaded.dt_old = 1.111_111_11e7;
    let exported = write_mod_string(&loaded).expect("write_mod_string");
    let ts_line = exported
        .lines()
        .find(|l| l.trim_start().starts_with("timestep (seconds)"))
        .expect("timestep (seconds) line present");
    assert!(
        ts_line.contains("9.87654321"),
        "expected model.dt in timestep (seconds), got: {ts_line}"
    );
    let reloaded =
        read_mod_str(&exported, loaded.controls.clone(), loaded.job.clone())
            .expect("round-trip exported .mod");
    assert!((reloaded.dt - loaded.dt).abs() <= 1.0e-15 * loaded.dt);

    // 5. Reject incomplete or negative `previous model` entries (`read_model.f90:505-516`).
    let missing_dt_next = "\n\
        previous model\n\
               previous n_shells                           2\n\
           previous mass (grams)      1.9884000000000000E+33\n\
              timestep (seconds)      4.3210000000000000E+09\n";
    let bad_prev_mod = minimal_two_zone_mod(4, "", missing_dt_next);
    assert!(read_mod_str(
        &bad_prev_mod,
        StarControls::default(),
        StarJobControls::default()
    )
    .is_err());

    let neg_mass_mod = minimal_two_zone_mod(
        4,
        "",
        "\n\
        previous model\n\
               previous n_shells                           2\n\
           previous mass (grams)     -1.0000000000000000E+00\n\
              timestep (seconds)      4.3210000000000000E+09\n\
               dt_next (seconds)      6.5430000000000000E+09\n",
    );
    assert!(read_mod_str(
        &neg_mass_mod,
        StarControls::default(),
        StarJobControls::default()
    )
    .is_err());
}

#[test]
fn photo_restart_preserves_checkpoint_z_flags_net_and_unclipped_dt() {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.01,
            zbase: 0.01,
            v_flag: true,
            rotation_flag: true,
            ..StarControls::default()
        },
        StarJobControls {
            net_name: String::from("o18_and_ne22.net"),
            ..StarJobControls::default()
        },
    )
    .expect("load zams");
    // The loaded ZAMS model is on the catalog's net (`init_model.f90:185`);
    // the checkpointed net name below stands for a later `change_net`.
    model.job.net_name = String::from("o18_and_ne22.net");
    model.dt_next = 0.0;
    model.aux.step_hist.dt_next_unclipped = 1.234e9;
    model.aux.step_hist.using_revised_max_yr_dt = true;
    model.aux.step_hist.revised_max_yr_dt = 42.5;
    model.aux.need_to_setvars = false;

    let photo = write_photo_bytes(&model);

    // Lossless checkpoint load preserves exact serialized bits/checksum.
    let loaded = load_photo_bytes(&photo).expect("load_photo_bytes");
    assert_eq!(
        compute_photo_checksum(&write_photo_bytes(&loaded)),
        compute_photo_checksum(&photo)
    );

    // Restart load with fresh controls applies non-checkpoint overrides
    // while preserving photo-authoritative state (`photo_in.f90:71-125,174,291`).
    let restart_controls = StarControls {
        initial_z: 0.03,
        v_flag: false,
        rotation_flag: false,
        mixing_length_alpha: 1.75,
        ..StarControls::default()
    };
    let restart_job = StarJobControls {
        net_name: String::from("pp_extras.net"),
        save_model_when_terminate: true,
        ..StarJobControls::default()
    };
    let restarted = read_photo_bytes(&photo, restart_controls, restart_job)
        .expect("read_photo_bytes");

    assert_eq!(
        restarted.controls.mixing_length_alpha.to_bits(),
        1.75_f64.to_bits()
    );
    assert!(restarted.job.save_model_when_terminate);
    assert_eq!(restarted.controls.initial_z.to_bits(), 0.01_f64.to_bits());
    assert!(restarted.controls.v_flag);
    assert!(restarted.controls.rotation_flag);
    assert_eq!(restarted.job.net_name, "o18_and_ne22.net");
    assert_eq!(
        restarted.controls.max_years_for_timestep.to_bits(),
        42.5_f64.to_bits()
    );
    assert_eq!(restarted.dt_next.to_bits(), 1.234e9_f64.to_bits());
    assert!(restarted.aux.need_to_setvars);
}

#[test]
fn primary_state_buffer_restores_generation_st_start_and_rollback_flags() {
    let mut model = load_zams_model(
        StarControls {
            initial_mass: 1.0,
            initial_z: 0.02,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    let nz = model.nz;

    model.model_number_old = 41;
    model.generations = 2;
    model.xmstar_old = 1.5e33;
    model.k_below_just_added = 7;
    model.surf_lns = 2.25e8;
    model.num_surf_revisions = 3;
    model.used_extra_iter_in_solver_for_accretion = true;
    model.using_gold_tolerances = true;
    model.why_tlim_neg_x = true;
    model.solver_skip_global_corr_coeff_limit = true;
    model.solver_first_try_of_step = false;
    model.adjust_j_q = 0.875;
    model.aux.have_j_rot = true;
    model.aux.need_to_setvars = false;
    model.aux.doing_finish_load_model = true;
    model.aux.st_start = StStartInfo {
        d_st_start: vec![123.0; nz],
        nu_st_start: vec![456.0; nz],
        have_st_start_info: true,
        prev_mesh_d_st_start: vec![789.0; nz],
        prev_mesh_nu_st_start: vec![321.0; nz],
        prev_mesh_dq: vec![0.1; nz],
        prev_mesh_nz: nz,
        prev_mesh_have_st_start_info: true,
    };
    model.aux.rsp.uq_rsp = vec![9.5; nz];
    model.aux.rsp.map_col_names = vec![String::from("r_cm"), String::from("v")];

    let buf = PrimaryStateBuffer::from_model(&model);
    assert!(buf.heap_bytes() > nz * 4 * std::mem::size_of::<f64>());

    // Clobber rollback fields and bump cumulative counters before restore.
    model.model_number_old = 99;
    model.generations = 1;
    model.xmstar_old = 0.0;
    model.k_below_just_added = 1;
    model.surf_lns = 0.0;
    model.num_surf_revisions = 0;
    model.used_extra_iter_in_solver_for_accretion = false;
    model.using_gold_tolerances = false;
    model.why_tlim_neg_x = false;
    model.solver_skip_global_corr_coeff_limit = false;
    model.solver_first_try_of_step = true;
    model.adjust_j_q = 1.0;
    model.aux.have_j_rot = false;
    model.aux.need_to_setvars = true;
    model.aux.doing_finish_load_model = false;
    model.aux.st_start = StStartInfo::default();
    model.aux.rsp.uq_rsp.clear();
    model.aux.rsp.map_col_names.clear();
    model.num_retries += 2;
    model.total_num_solver_iterations += 11;
    model.step_retry_cnt = 4;

    buf.restore_into(&mut model).expect("restore_into");

    assert_eq!(model.model_number_old, 41);
    assert_eq!(model.generations, 2);
    assert_eq!(model.xmstar_old.to_bits(), 1.5e33_f64.to_bits());
    assert_eq!(model.k_below_just_added, 7);
    assert_eq!(model.surf_lns.to_bits(), 2.25e8_f64.to_bits());
    assert_eq!(model.num_surf_revisions, 3);
    assert!(model.used_extra_iter_in_solver_for_accretion);
    assert!(model.using_gold_tolerances);
    assert!(model.why_tlim_neg_x);
    assert!(model.solver_skip_global_corr_coeff_limit);
    assert!(!model.solver_first_try_of_step);
    assert_eq!(model.step_retry_cnt, 4);
    assert_eq!(model.adjust_j_q.to_bits(), 0.875_f64.to_bits());
    assert!(model.aux.have_j_rot);
    assert!(!model.aux.need_to_setvars);
    assert!(model.aux.doing_finish_load_model);
    assert!(model.aux.st_start.have_st_start_info);
    assert_eq!(model.aux.st_start.prev_mesh_nz, nz);
    assert_eq!(
        model.aux.st_start.d_st_start[0].to_bits(),
        123.0_f64.to_bits()
    );
    assert_eq!(
        model.aux.st_start.prev_mesh_nu_st_start[nz - 1].to_bits(),
        321.0_f64.to_bits()
    );
    assert_eq!(model.aux.rsp.uq_rsp[0].to_bits(), 9.5_f64.to_bits());
    assert_eq!(model.aux.rsp.map_col_names, vec!["r_cm", "v"]);
    assert_eq!(model.num_retries, 2);
    assert_eq!(model.total_num_solver_iterations, 11);
}

#[test]
fn rn_parser_handles_quotes_cp_flags_negated_and_nested_conditionals() {
    let dir = TempDirGuard::new("rn_parser");
    let script = "\
#!/usr/bin/env bash
do_one \"\" \"stage 1.mod\" ''
do_one 'inlist #2' final.mod # trailing comment
cp -f -- \"standard a.mod\" a.mod
do_one_binary \\
  inlist_bin bin.mod
if [ ! -z \"$MESA_SKIP_OPTIONAL\" ]; then
   cp -p opt_std.mod opt.mod
else
   if [ -f ignored_inner ]; then
      do_one inner_opt.mod
   fi
   do_one full_opt.inlist full_opt.mod; cp full_opt.mod saved.mod
fi
";
    fs::write(dir.path().join("rn"), script).expect("write synthetic rn");

    let common = vec![
        RnStage::DoOne {
            inlist: String::from("inlist"),
            output_mod: Some(String::from("stage 1.mod")),
            logs_dir: None,
        },
        RnStage::DoOne {
            inlist: String::from("inlist #2"),
            output_mod: Some(String::from("final.mod")),
            logs_dir: None,
        },
        RnStage::CopyCheckpoint {
            src_mod: String::from("standard a.mod"),
            dst_mod: String::from("a.mod"),
        },
        RnStage::DoOne {
            inlist: String::from("inlist_bin"),
            output_mod: Some(String::from("bin.mod")),
            logs_dir: None,
        },
    ];

    let mut expected_skip = common.clone();
    expected_skip.push(RnStage::CopyCheckpoint {
        src_mod: String::from("opt_std.mod"),
        dst_mod: String::from("opt.mod"),
    });
    assert_eq!(
        parse_rn_stages(dir.path(), true).expect("skip_optional=true"),
        expected_skip
    );

    let mut expected_full = common;
    expected_full.extend([
        RnStage::DoOne {
            inlist: String::from("inner_opt.mod"),
            output_mod: None,
            logs_dir: None,
        },
        RnStage::DoOne {
            inlist: String::from("full_opt.inlist"),
            output_mod: Some(String::from("full_opt.mod")),
            logs_dir: None,
        },
        RnStage::CopyCheckpoint {
            src_mod: String::from("full_opt.mod"),
            dst_mod: String::from("saved.mod"),
        },
    ]);
    assert_eq!(
        parse_rn_stages(dir.path(), false).expect("skip_optional=false"),
        expected_full
    );
}

#[test]
fn rn_parser_handles_elif_and_else_branches() {
    let dir = TempDirGuard::new("rn_parser_elif");
    let elif_script = "\
if [ -n \"$MESA_SKIP_OPTIONAL\" ]; then
   do_one branch_if
elif [ -z \"$MESA_RUN_OPTIONAL\" ]; then
   do_one branch_elif
else
   do_one branch_else
fi
";
    fs::write(dir.path().join("rn"), elif_script).expect("write elif rn");
    assert_eq!(
        parse_rn_stages(dir.path(), true).expect("elif skip"),
        vec![RnStage::DoOne {
            inlist: String::from("branch_if"),
            output_mod: None,
            logs_dir: None,
        }]
    );
    assert_eq!(
        parse_rn_stages(dir.path(), false).expect("elif full"),
        vec![RnStage::DoOne {
            inlist: String::from("branch_else"),
            output_mod: None,
            logs_dir: None,
        }]
    );
}
