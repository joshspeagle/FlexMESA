// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream Fortran MESA (release `26.4.1`, commit
//! `c8c202662922a22b3efefa824473ad615c623afd`) Cepheid 2nd-crossing
//! instability-strip boundary interpolation tests (`rsp_check_2nd_crossing`,
//! `#5`, `#141`).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use flexmesa_eos::EosConfig;
use flexmesa_interp1d::{interp_pm, interp_values, interpolate_vector_pm};
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};
use flexmesa_star::{
    do_rsp_build_model, StarControls, StarError, StarJobControls,
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rsp2ndCrossingPoint {
    pub model_number: i64,
    pub period_days: f64,
    pub growth_rate: f64,
    pub teff_k: f64,
    pub lum_lsun: f64,
    pub star_age_yr: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rsp2ndCrossingOffset {
    pub offset_k: i64,
    pub teff_k: f64,
    pub lum_lsun: f64,
}

#[derive(Debug, Clone)]
struct Rsp2ndCrossingReport {
    pub teff_blue_k: f64,
    pub teff_red_k: f64,
    pub points: Vec<Rsp2ndCrossingPoint>,
    pub offsets: Vec<Rsp2ndCrossingOffset>,
}

/// Ported test-side implementation of `star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:58-371`.
fn eval_rsp_2nd_crossing_with_controls(
    case_dir: &Path,
    base_controls: &StarControls,
    base_job: &StarJobControls,
    eos_config: &EosConfig,
    kap_config: &KapConfig,
    history_filename: &str,
) -> Result<Rsp2ndCrossingReport, StarError> {
    let xc = &base_controls.x_ctrl;
    let xic = &base_controls.x_integer_ctrl;
    let delta_teff = xc.first().copied().filter(|&v| v > 0.0).unwrap_or(100.0);
    let log_t3 = xc.get(5).copied().filter(|&v| v > 0.0).unwrap_or(3.65);
    let log_l3 = xc.get(6).copied().filter(|&v| v > 0.0).unwrap_or(4.5);
    let log_t4 = xc.get(7).copied().filter(|&v| v > 0.0).unwrap_or(3.77);
    let log_l4 = xc.get(8).copied().filter(|&v| v > 0.0).unwrap_or(2.4);

    let skip_cols = xic.first().copied().unwrap_or(6).max(0) as usize;
    let col_mod = xic.get(1).copied().unwrap_or(1).max(1) as usize - 1;
    let col_age = xic.get(2).copied().unwrap_or(2).max(1) as usize - 1;
    let col_log_t = xic.get(3).copied().unwrap_or(4).max(1) as usize - 1;
    let col_log_l = xic.get(4).copied().unwrap_or(5).max(1) as usize - 1;
    let max_col = col_mod.max(col_age).max(col_log_t).max(col_log_l);

    let fname = if history_filename.trim().is_empty() {
        "history_7M.data"
    } else {
        history_filename.trim()
    };
    let path = case_dir.join(fname);
    let text = fs::read_to_string(&path).map_err(|e| {
        StarError::IoError(format!("failed to read {}: {e}", path.display()))
    })?;

    let get_red_log_t = |log_l: f64| -> f64 {
        log_t4 + (log_l - log_l4) * (log_t3 - log_t4) / (log_l3 - log_l4)
    };

    let mut finished_1st = false;
    let mut in_2nd = false;
    let mut have_first = false;
    let mut just_failed = false;
    let mut num_beyond_blue = 0_usize;
    let mut prev_teff = 1.0e99_f64;
    let mut points = Vec::new();

    for line in text.lines().skip(skip_cols) {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() <= max_col {
            continue;
        }
        let (Ok(m_f), Ok(age), Ok(lt), Ok(ll)) = (
            toks[col_mod].parse::<f64>(),
            toks[col_age].parse::<f64>(),
            toks[col_log_t].parse::<f64>(),
            toks[col_log_l].parse::<f64>(),
        ) else {
            continue;
        };
        let mod_num = m_f as i64;
        let teff = flexmesa_math::exp10(lt);
        let lum = flexmesa_math::exp10(ll);

        if !finished_1st {
            let min_t = flexmesa_math::exp10(get_red_log_t(ll));
            finished_1st = teff < min_t;
            prev_teff = teff;
            continue;
        }
        if teff < prev_teff {
            prev_teff = teff;
            continue;
        }
        let min_t = flexmesa_math::exp10(get_red_log_t(ll));
        if teff < min_t - 4.0 * delta_teff {
            prev_teff = teff;
            continue;
        }
        if have_first && teff - delta_teff < prev_teff {
            continue;
        }

        let mut ctrl = base_controls.clone();
        ctrl.rsp_teff = teff;
        ctrl.rsp_l = lum;
        ctrl.rsp.nmodes = 1;
        ctrl.rsp.use_other_rsp_build_model = false;
        ctrl.rsp.use_other_rsp_linear_analysis = false;
        ctrl.rsp.use_rsp_new_start_scheme = false;

        let Ok((_m, Some(lina))) = do_rsp_build_model(
            ctrl,
            base_job.clone(),
            *eos_config,
            kap_config.clone(),
            None,
        ) else {
            if num_beyond_blue > 0 || just_failed {
                break;
            }
            just_failed = true;
            continue;
        };
        just_failed = false;

        let period_days = lina.periods_sec[0] / 86_400.0;
        let growth = lina.growth_rates[0];
        if !have_first && growth > 0.0 {
            return Err(StarError::ConvergenceFailure(String::from(
                "rsp_check_2nd_crossing: first sampled point already unstable",
            )));
        }
        have_first = true;
        if growth > 0.0 {
            in_2nd = true;
        }
        if growth <= 0.0 && in_2nd {
            num_beyond_blue += 1;
        }
        points.push(Rsp2ndCrossingPoint {
            model_number: mod_num,
            period_days,
            growth_rate: growth,
            teff_k: teff,
            lum_lsun: lum,
            star_age_yr: age,
        });
        if num_beyond_blue == 2 || points.len() == 200 {
            break;
        }
        prev_teff = teff;
    }

    let n = points.len();
    if n < 3 || points[0].growth_rate >= 0.0 || num_beyond_blue < 1 {
        return Err(StarError::ConvergenceFailure(format!(
            "rsp_check_2nd_crossing: insufficient strip coverage (n={n}, beyond_blue={num_beyond_blue})"
        )));
    }

    let growth: Vec<f64> = points.iter().map(|p| p.growth_rate).collect();
    let temp: Vec<f64> = points.iter().map(|p| p.teff_k).collect();
    let lum: Vec<f64> = points.iter().map(|p| p.lum_lsun).collect();

    let teff_red = interpolate_red_edge(&growth, &temp)?;
    let teff_blue = interpolate_blue_edge(&growth, &temp, num_beyond_blue)?;

    let lum_coeffs = interp_pm(&temp, &lum).map_err(|e| {
        StarError::ConvergenceFailure(format!("L(Teff) interp_pm failed: {e}"))
    })?;
    let mut offsets = Vec::new();
    let mut offset = 0.0_f64;
    loop {
        let x_val = (teff_blue - offset).max(teff_red);
        let y_val =
            interp_values(&temp, &lum_coeffs, &[x_val]).map_err(|e| {
                StarError::ConvergenceFailure(format!(
                    "L(Teff) interp_values failed: {e}"
                ))
            })?;
        offsets.push(Rsp2ndCrossingOffset {
            offset_k: (teff_blue - x_val) as i64,
            teff_k: x_val,
            lum_lsun: y_val[0],
        });
        if x_val == teff_red {
            break;
        }
        offset += 100.0;
    }

    if let Some(out_name) = base_controls
        .x_character_ctrl
        .get(1)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        write_2nd_crossing_file(
            &case_dir.join(out_name),
            base_controls,
            &points,
            teff_red,
            teff_blue,
            &offsets,
        )?;
    }

    Ok(Rsp2ndCrossingReport {
        teff_blue_k: teff_blue,
        teff_red_k: teff_red,
        points,
        offsets,
    })
}

fn interpolate_red_edge(
    growth: &[f64],
    temp: &[f64],
) -> Result<f64, StarError> {
    let n = growth.len();
    if growth[1] < 0.0 {
        for i in 3..n {
            if growth[i] > 0.0
                && growth[i - 1] > 0.0
                && growth[i - 2] < 0.0
                && growth[i - 3] < 0.0
            {
                let v = interpolate_vector_pm(
                    &growth[i - 3..=i],
                    &temp[i - 3..=i],
                    &[0.0],
                )
                .map_err(|e| {
                    StarError::ConvergenceFailure(format!(
                        "red-edge Steffen interp failed: {e}"
                    ))
                })?;
                return Ok(v[0]);
            }
        }
    } else if growth[2] > 0.0 {
        let v = interpolate_vector_pm(&growth[0..3], &temp[0..3], &[0.0])
            .map_err(|e| {
                StarError::ConvergenceFailure(format!(
                    "red-edge 3-pt Steffen interp failed: {e}"
                ))
            })?;
        return Ok(v[0]);
    }
    Err(StarError::ConvergenceFailure(String::from(
        "rsp_check_2nd_crossing: failed to locate Teff_red_edge",
    )))
}

fn interpolate_blue_edge(
    growth: &[f64],
    temp: &[f64],
    num_beyond_blue: usize,
) -> Result<f64, StarError> {
    let n = growth.len();
    if num_beyond_blue == 2 && n >= 4 {
        for i in (0..=n - 4).rev() {
            if growth[i] > 0.0
                && growth[i + 1] > 0.0
                && growth[i + 2] < 0.0
                && growth[i + 3] < 0.0
            {
                let v = interpolate_vector_pm(
                    &growth[i..=i + 3],
                    &temp[i..=i + 3],
                    &[0.0],
                )
                .map_err(|e| {
                    StarError::ConvergenceFailure(format!(
                        "blue-edge Steffen interp failed: {e}"
                    ))
                })?;
                return Ok(v[0]);
            }
        }
    } else if num_beyond_blue == 1
        && n > 3
        && growth[n - 1] < 0.0
        && growth[n - 2] > 0.0
        && growth[n - 3] > 0.0
    {
        let i = n - 3;
        let v =
            interpolate_vector_pm(&growth[i..=i + 2], &temp[i..=i + 2], &[0.0])
                .map_err(|e| {
                    StarError::ConvergenceFailure(format!(
                        "blue-edge 3-pt Steffen interp failed: {e}"
                    ))
                })?;
        return Ok(v[0]);
    }
    Err(StarError::ConvergenceFailure(String::from(
        "rsp_check_2nd_crossing: failed to locate Teff_blue_edge",
    )))
}

fn write_2nd_crossing_file(
    out_path: &Path,
    ctrl: &StarControls,
    points: &[Rsp2ndCrossingPoint],
    teff_red: f64,
    teff_blue: f64,
    offsets: &[Rsp2ndCrossingOffset],
) -> Result<(), StarError> {
    let mut buf = String::with_capacity(4096);
    for (k, v) in [
        ("RSP_mass", ctrl.rsp_mass),
        ("RSP_X", ctrl.rsp_x),
        ("RSP_Z", ctrl.rsp_z),
        ("RSP_alfam", ctrl.rsp.alfam),
        ("RSP_alfap", ctrl.rsp.alfap),
        ("RSP_alfat", ctrl.rsp.alfat),
        ("RSP_gammar", ctrl.rsp.gammar),
    ] {
        let _ = writeln!(buf, "{k:>24}{v:26.16E}");
    }
    let _ = writeln!(buf);
    let _ = writeln!(
        buf,
        "{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}",
        "model_number", "period(d)", "growth", "Teff", "L", "star_age"
    );
    for p in points {
        let _ = writeln!(
            buf,
            "{:20}{:20.10E}{:20.10E}{:20.10E}{:20.10E}{:20.10E}",
            p.model_number,
            p.period_days,
            p.growth_rate,
            p.teff_k,
            p.lum_lsun,
            p.star_age_yr
        );
    }
    let _ = writeln!(buf, "\n{:>20}{teff_red:20.10}", "Teff_red_edge");
    let _ = writeln!(buf, "{:>20}{teff_blue:20.10}\n", "Teff_blue_edge");
    let _ = writeln!(buf, "{:>10}{:>20}{:>20}", "offset", "Teff", "L");
    for o in offsets {
        let _ = writeln!(
            buf,
            "{:10}{:20.10}{:20.10}",
            o.offset_k, o.teff_k, o.lum_lsun
        );
    }
    let _ = writeln!(buf);
    fs::write(out_path, buf).map_err(|e| {
        StarError::IoError(format!(
            "failed to write {}: {e}",
            out_path.display()
        ))
    })
}

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
