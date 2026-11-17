// SPDX-License-Identifier: LGPL-3.0-or-later
//! Upstream Fortran MESA (release `26.4.1`, commit
//! `c8c202662922a22b3efefa824473ad615c623afd`) Cepheid 2nd-crossing
//! instability-strip boundary interpolation tests (`rsp_check_2nd_crossing`,
//! `#5`, `#141`).

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_eos::EosConfig;
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_star::{
    eval_rsp_2nd_crossing_with_controls, StarControls, StarJobControls,
};

fn golden_rsp_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("tests")
        .join("fixtures")
        .join("golden")
        .join("rsp")
}

fn a09_kap_config(zbase: f64) -> KapConfig {
    KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase,
        ..Default::default()
    }
}

fn base_rsp_job() -> StarJobControls {
    StarJobControls {
        create_rsp_model: true,
        initial_zfracs: 6,
        ..Default::default()
    }
}

type CrossingPoint = (i64, f64, f64, f64, f64);

fn load_crossing_5m_26_4_1(path: &Path) -> (Vec<CrossingPoint>, f64, f64) {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        raw.contains("# mesa_version_number=r26.4.1"),
        "missing r26.4.1 header in {}",
        path.display()
    );
    let mut pts = Vec::new();
    let mut teff_red = None;
    let mut teff_blue = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        match cols[0] {
            "PT" => {
                assert_eq!(cols.len(), 6, "unexpected PT row: {trimmed}");
                pts.push((
                    cols[1].parse::<i64>().expect("modnum"),
                    cols[2].parse::<f64>().expect("teff"),
                    cols[3].parse::<f64>().expect("lum"),
                    cols[4].parse::<f64>().expect("period_days"),
                    cols[5].parse::<f64>().expect("growth_rate"),
                ));
            }
            "EDGE" => {
                assert_eq!(cols.len(), 3, "unexpected EDGE row: {trimmed}");
                let val = cols[2].parse::<f64>().expect("edge val");
                match cols[1] {
                    "Teff_red_edge" => teff_red = Some(val),
                    "Teff_blue_edge" => teff_blue = Some(val),
                    other => panic!("unexpected edge name: {other}"),
                }
            }
            other => panic!("unexpected row prefix: {other}"),
        }
    }
    (
        pts,
        teff_red.expect("Teff_red_edge"),
        teff_blue.expect("Teff_blue_edge"),
    )
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_rsp_lina_and_steffen_edges_match_mesa_26_4_1_5m_reference() {
    // 13 sampled 2nd-crossing points up to `num_beyond_blue_edge == 2`
    // (`star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:223` in
    // MESA release `26.4.1`), parsed from `crossing_5m_26_4_1.tsv` alongside
    // historical MESA `v11461` (`OPAL` EOS) `list` values.
    let (ref_pts, ref_teff_red_k, ref_teff_blue_k) = load_crossing_5m_26_4_1(
        &golden_rsp_dir().join("crossing_5m_26_4_1.tsv"),
    );
    let legacy_v11461: [(f64, f64); 13] = [
        (6.9798, -1.4649e-2),
        (6.5072, -9.8892e-3),
        (6.0527, -4.869_439_213_642_279e-3),
        (5.6574, -2.440_425_682_565_504e-4),
        (5.3266, 4.045_355_766_436_185e-3),
        (5.0226, 8.141_442_149_700_16e-3),
        (4.7364, 1.1213e-2),
        (4.4683, 1.2694e-2),
        (4.2169, 1.1268e-2),
        (4.0071, 8.689_902_602_617_62e-3),
        (3.8078, 4.316_245_731_944_837e-3),
        (3.6174, -2.786_712_154_970_167_4e-4),
        (3.4361, -5.184_950_937_238_88e-3),
    ];
    assert_eq!(ref_pts.len(), 13);

    let tmp_dir = std::env::temp_dir()
        .join(format!("flexmesa_rsp_5m_crossing_{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let mut hist = String::from(
        "h1\nh2\nh3\nh4\nh5\nmodel_number star_age star_mass log_Teff log_L\n",
    );
    hist.push_str("2100 5.0e7 5.0 3.6800000000000000 3.1600000000000000\n");
    for (idx, &(modnum, teff, lum, _, _)) in ref_pts.iter().enumerate() {
        let idx_f64 = f64::from(u32::try_from(idx).expect("idx fits in u32"));
        let _ = writeln!(
            hist,
            "{modnum} {:.6e} 5.0 {:.16e} {:.16e}",
            5.1e7 + idx_f64 * 1.0e4,
            flexmesa_math::log10(teff),
            flexmesa_math::log10(lum)
        );
    }
    fs::write(tmp_dir.join("history_5M.data"), &hist).expect("write history");

    let mut ctrl = StarControls {
        rsp_mass: 5.0,
        rsp_x: 0.736,
        rsp_z: 0.008,
        zbase: 0.008,
        ..Default::default()
    };
    ctrl.rsp.t_anchor_tolerance = 1.0e-6;
    ctrl.rsp.t_inner_tolerance = 1.0e-5;
    ctrl.x_ctrl[0] = 100.0;
    ctrl.x_ctrl[1] = 3.76;
    ctrl.x_ctrl[2] = 4.5;
    ctrl.x_ctrl[3] = 3.83;
    ctrl.x_ctrl[4] = 2.4;
    ctrl.x_ctrl[5] = 3.65;
    ctrl.x_ctrl[6] = 4.5;
    ctrl.x_ctrl[7] = 3.77;
    ctrl.x_ctrl[8] = 2.4;
    ctrl.x_integer_ctrl[0] = 6;
    ctrl.x_integer_ctrl[1] = 1;
    ctrl.x_integer_ctrl[2] = 2;
    ctrl.x_integer_ctrl[3] = 4;
    ctrl.x_integer_ctrl[4] = 5;
    ctrl.x_character_ctrl[1] = String::from("blue_edge_5M_A.data");

    let rep = eval_rsp_2nd_crossing_with_controls(
        &tmp_dir,
        &ctrl,
        &base_rsp_job(),
        &EosConfig::default(),
        &a09_kap_config(0.008),
        "history_5M.data",
    )
    .expect("eval_rsp_2nd_crossing_with_controls");

    assert_eq!(rep.points.len(), 13);
    for ((p, &(_m, _t, _l, p_ref, g_ref)), &(p_v11461, g_v11461)) in
        rep.points.iter().zip(ref_pts.iter()).zip(&legacy_v11461)
    {
        let rel_p_ref = (p.period_days - p_ref).abs() / p_ref;
        let abs_g_ref = (p.growth_rate - g_ref).abs();
        let rel_p_legacy = (p.period_days - p_v11461).abs() / p_v11461;
        let abs_g_legacy = (p.growth_rate - g_v11461).abs();
        assert!(
            rel_p_ref < 1.0e-6,
            "mod {}: P_days={} vs 26.4.1 {p_ref} (rel={rel_p_ref:.3e})",
            p.model_number,
            p.period_days
        );
        assert!(
            abs_g_ref < 5.0e-6,
            "mod {}: growth={} vs 26.4.1 {g_ref} (diff={abs_g_ref:.3e})",
            p.model_number,
            p.growth_rate
        );
        assert!(rel_p_legacy < 3.0e-3);
        assert!(abs_g_legacy < 5.0e-4);
    }

    assert!(
        (rep.teff_red_k - ref_teff_red_k).abs() < 5.0,
        "Teff_red_k={} vs 26.4.1 {ref_teff_red_k}",
        rep.teff_red_k
    );
    assert!(
        (rep.teff_blue_k - ref_teff_blue_k).abs() < 5.0,
        "Teff_blue_k={} vs 26.4.1 {ref_teff_blue_k}",
        rep.teff_blue_k
    );
    assert_eq!(rep.offsets.len(), 10);
    assert!(tmp_dir.join("blue_edge_5M_A.data").is_file());

    let _ = fs::remove_file(tmp_dir.join("history_5M.data"));
    let _ = fs::remove_file(tmp_dir.join("blue_edge_5M_A.data"));
    let _ = fs::remove_dir(&tmp_dir);
}

#[test]
fn test_rsp_check_2nd_crossing_on_7m_mesa_test_suite() {
    let rel = "star/test_suite/rsp_check_2nd_crossing/history_logs/history_7M.data";
    let Some(hist_path) = flexmesa_utils::require_mesa_test_input_path(rel) else {
        return;
    };
    let case_dir = hist_path
        .parent()
        .and_then(|p| p.parent())
        .expect("rsp_check_2nd_crossing case directory")
        .to_path_buf();
    let hist_rel = "history_logs/history_7M.data";

    let ref_7b = fs::read_to_string(
        golden_rsp_dir().join("7B_2nd_crossing_26_4_1.data"),
    )
    .expect("read 7B_2nd_crossing_26_4_1.data");
    let mut ref_7m_red = None;
    let mut ref_7m_blue = None;
    for line in ref_7b.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() == 2 && cols[0] == "Teff_red_edge" {
            ref_7m_red = Some(cols[1].parse::<f64>().expect("7M red"));
        } else if cols.len() == 2 && cols[0] == "Teff_blue_edge" {
            ref_7m_blue = Some(cols[1].parse::<f64>().expect("7M blue"));
        }
    }
    let expected_red = ref_7m_red.expect("Teff_red_edge in 7B");
    let expected_blue = ref_7m_blue.expect("Teff_blue_edge in 7B");

    let mut ctrl = StarControls {
        rsp_mass: 7.0,
        rsp_x: 0.736,
        rsp_z: 0.008,
        zbase: 0.008,
        ..Default::default()
    };
    ctrl.rsp.alfam = 0.5;
    ctrl.rsp.gammar = 1.0;
    ctrl.rsp.t_anchor_tolerance = 1.0e-6;
    ctrl.rsp.t_inner_tolerance = 1.0e-5;
    ctrl.x_ctrl[0] = 100.0;
    ctrl.x_ctrl[1] = 3.76;
    ctrl.x_ctrl[2] = 4.5;
    ctrl.x_ctrl[3] = 3.83;
    ctrl.x_ctrl[4] = 2.4;
    ctrl.x_ctrl[5] = 3.65;
    ctrl.x_ctrl[6] = 4.5;
    ctrl.x_ctrl[7] = 3.77;
    ctrl.x_ctrl[8] = 2.4;
    ctrl.x_integer_ctrl[0] = 6;
    ctrl.x_integer_ctrl[1] = 1;
    ctrl.x_integer_ctrl[2] = 2;
    ctrl.x_integer_ctrl[3] = 4;
    ctrl.x_integer_ctrl[4] = 5;

    let rep = eval_rsp_2nd_crossing_with_controls(
        &case_dir,
        &ctrl,
        &base_rsp_job(),
        &EosConfig::default(),
        &a09_kap_config(0.008),
        hist_rel,
    )
    .expect("7M rsp_check_2nd_crossing");

    assert_eq!(rep.points.len(), 16);
    assert!(
        (rep.teff_red_k - expected_red).abs() < 5.0,
        "7M Teff_red_k={} vs 26.4.1 {expected_red}",
        rep.teff_red_k
    );
    assert!(
        (rep.teff_blue_k - expected_blue).abs() < 5.0,
        "7M Teff_blue_k={} vs 26.4.1 {expected_blue}",
        rep.teff_blue_k
    );
    assert_eq!(rep.offsets.len(), 11);
}
