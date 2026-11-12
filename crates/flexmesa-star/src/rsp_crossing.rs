// SPDX-License-Identifier: LGPL-3.0-or-later
//! Cepheid 2nd-crossing instability-strip boundary finder
//! (`rsp_check_2nd_crossing` in
//! `star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:58-371`).

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use flexmesa_eos::EosConfig;
use flexmesa_interp1d::{interp_pm, interp_values, interpolate_vector_pm};
use flexmesa_kap::{KapCoOption, KapConfig, KapLowTOption, KapOption};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::rsp_build::do_rsp_build_model;

/// Single evolutionary track point evaluated during a Cepheid 2nd-crossing
/// linear stability scan (`rsp_check_2nd_crossing`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rsp2ndCrossingPoint {
    /// Evolutionary track `model_number`.
    pub model_number: i64,
    /// Fundamental-mode period $P_0$ in days.
    pub period_days: f64,
    /// Fundamental-mode fractional growth rate $\eta_0$.
    pub growth_rate: f64,
    /// Effective temperature $T_{\mathrm{eff}}$ in $\mathrm{K}$.
    pub teff_k: f64,
    /// Luminosity $L / L_\odot$.
    pub lum_lsun: f64,
    /// Stellar age in years.
    pub star_age_yr: f64,
}

/// Interpolated $(T_{\mathrm{eff}}, L/L_\odot)$ point spaced inward from the
/// blue edge across the 2nd-crossing instability strip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rsp2ndCrossingOffset {
    /// Integer temperature offset $\lfloor T_{\mathrm{eff,blue}} - T_{\mathrm{eff}} \rfloor$ ($\mathrm{K}$).
    pub offset_k: i64,
    /// Effective temperature $T_{\mathrm{eff}}$ ($\mathrm{K}$).
    pub teff_k: f64,
    /// Steffen-interpolated luminosity $L / L_\odot$.
    pub lum_lsun: f64,
}

/// Full Cepheid 2nd-crossing instability-strip report (`rsp_check_2nd_crossing`).
#[derive(Debug, Clone)]
pub struct Rsp2ndCrossingReport {
    /// Blue-edge effective temperature $T_{\mathrm{eff,blue}}$ ($\mathrm{K}$).
    pub teff_blue_k: f64,
    /// Red-edge effective temperature $T_{\mathrm{eff,red}}$ ($\mathrm{K}$).
    pub teff_red_k: f64,
    /// Instability-strip width $T_{\mathrm{eff,blue}} - T_{\mathrm{eff,red}}$ ($\mathrm{K}$).
    pub delta_teff_k: f64,
    /// Sampled 2nd-crossing models and their `LINA` F-mode periods/growth rates.
    pub points: Vec<Rsp2ndCrossingPoint>,
    /// Steffen-interpolated `L(Teff)` grid across `[Teff_red, Teff_blue]`.
    pub offsets: Vec<Rsp2ndCrossingOffset>,
}

/// Evaluates a full Cepheid 2nd-crossing linear stability scan using explicit
/// `StarControls`, `StarJobControls`, `EosConfig`, and `KapConfig`
/// (`star/test_suite/rsp_check_2nd_crossing/src/run_star_extras.f90:58-371`).
///
/// # Errors
/// Returns [`StarError`] if the evolutionary history file cannot be read or
/// red/blue instability-strip edges cannot be bracketed.
pub fn eval_rsp_2nd_crossing_with_controls(
    case_dir: &Path,
    base_controls: &StarControls,
    base_job: &StarJobControls,
    eos_config: &EosConfig,
    kap_config: &KapConfig,
    history_filename: &str,
) -> Result<Rsp2ndCrossingReport, StarError> {
    let xc = &base_controls.x_ctrl;
    let xic = &base_controls.x_integer_ctrl;
    let delta_teff = xc.get(1).copied().filter(|&v| v > 0.0).unwrap_or(100.0);
    let log_t1 = xc.get(2).copied().filter(|&v| v > 0.0).unwrap_or(3.76);
    let log_l1 = xc.get(3).copied().filter(|&v| v > 0.0).unwrap_or(4.5);
    let log_t2 = xc.get(4).copied().filter(|&v| v > 0.0).unwrap_or(3.83);
    let log_l2 = xc.get(5).copied().filter(|&v| v > 0.0).unwrap_or(2.4);
    let log_t3 = xc.get(6).copied().filter(|&v| v > 0.0).unwrap_or(3.65);
    let log_l3 = xc.get(7).copied().filter(|&v| v > 0.0).unwrap_or(4.5);
    let log_t4 = xc.get(8).copied().filter(|&v| v > 0.0).unwrap_or(3.77);
    let log_l4 = xc.get(9).copied().filter(|&v| v > 0.0).unwrap_or(2.4);
    let _ = (log_t1, log_l1, log_t2, log_l2);

    let skip_cols = xic.get(1).copied().unwrap_or(6).max(0) as usize;
    let col_mod = xic.get(2).copied().unwrap_or(1).max(1) as usize - 1;
    let col_age = xic.get(3).copied().unwrap_or(2).max(1) as usize - 1;
    let col_log_t = xic.get(4).copied().unwrap_or(4).max(1) as usize - 1;
    let col_log_l = xic.get(5).copied().unwrap_or(5).max(1) as usize - 1;
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
        let teff = 10.0_f64.powf(lt);
        let lum = 10.0_f64.powf(ll);

        if !finished_1st {
            let min_t = 10.0_f64.powf(get_red_log_t(ll));
            finished_1st = teff < min_t;
            prev_teff = teff;
            continue;
        }
        if teff < prev_teff {
            prev_teff = teff;
            continue;
        }
        let min_t = 10.0_f64.powf(get_red_log_t(ll));
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
        .get(2)
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
        delta_teff_k: teff_blue - teff_red,
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

/// Evaluates `(Teff_blue, Teff_red, delta_Teff)` across a Cepheid 2nd-crossing
/// evolutionary track using standard `7 Msun`, `X = 0.736`, `Z = 0.008` Set B
/// `RSP` controls (`star/test_suite/rsp_check_2nd_crossing`).
///
/// # Errors
/// Returns [`StarError`] if `history_filename` cannot be read or edges fail.
pub fn eval_rsp_2nd_crossing_edges(
    case_dir: &Path,
    x_ctrl: &[f64],
    x_integer_ctrl: &[i64],
    history_filename: &str,
) -> Result<(f64, f64, f64), StarError> {
    let mut controls = StarControls {
        rsp_mass: 7.0,
        rsp_x: 0.736,
        rsp_z: 0.008,
        zbase: 0.008,
        x_ctrl: x_ctrl.to_vec(),
        x_integer_ctrl: x_integer_ctrl.to_vec(),
        ..Default::default()
    };
    controls.rsp.alfam = 0.5;
    controls.rsp.gammar = 1.0;
    controls.rsp.t_anchor_tolerance = 1.0e-6;
    controls.rsp.t_inner_tolerance = 1.0e-5;

    let job = StarJobControls {
        create_rsp_model: true,
        initial_zfracs: 6,
        ..Default::default()
    };
    let eos_config = EosConfig::default();
    let kap_config = KapConfig {
        kap_option: KapOption::A09,
        kap_co_option: KapCoOption::A09Co,
        kap_low_t_option: KapLowTOption::LowTFa05A09p,
        zbase: 0.008,
        ..Default::default()
    };

    let rep = eval_rsp_2nd_crossing_with_controls(
        case_dir,
        &controls,
        &job,
        &eos_config,
        &kap_config,
        history_filename,
    )?;
    Ok((rep.teff_blue_k, rep.teff_red_k, rep.delta_teff_k))
}
