// SPDX-License-Identifier: LGPL-3.0-or-later
//! Multi-star worker-pool batching experiment (`N` independent stars evolved
//! concurrently on a `rayon::ThreadPool`, `#168` Task 6).

use std::fs;
use std::path::Path;
use std::time::Instant;

use flexmesa_math::numerics_mode;
use flexmesa_star::evolve::star_evolve_step_with_extras;
use flexmesa_star::history::extract_history_row;
use flexmesa_star::hydro_vars::set_hydro_vars;
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::DefaultStarExtras;
use flexmesa_star::model::StarModel;
use flexmesa_star::rn_parser::{parse_rn_stages, RnStage};
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;
use flexmesa_star::termination::check_termination_limits_with_history;
use rayon::prelude::*;

use crate::json_fmt::{
    current_rss_kib, peak_rss_kib, JsonObjectBuilder, StagedCaseGuard,
};
use crate::step_profiler::warmup_static_tables;

/// Single operating-point result for `n_stars` evolved together on `pool_threads`.
#[derive(Debug, Clone)]
pub struct BatchPointRecord {
    pub n_stars: usize,
    pub pool_threads: usize,
    pub wall_sec: f64,
    pub total_steps: usize,
    pub total_zone_steps: usize,
    pub stars_per_min: f64,
    pub steps_per_sec: f64,
    pub zone_steps_per_sec: f64,
    pub speedup_vs_n1: f64,
    pub current_rss_kib: usize,
    pub peak_rss_kib: usize,
}

impl BatchPointRecord {
    #[must_use]
    pub fn to_json(&self, indent: usize) -> String {
        let mut obj = JsonObjectBuilder::new(indent);
        obj.field_usize("n_stars", self.n_stars);
        obj.field_usize("pool_threads", self.pool_threads);
        obj.field_f64("wall_sec", self.wall_sec);
        obj.field_usize("total_steps", self.total_steps);
        obj.field_usize("total_zone_steps", self.total_zone_steps);
        obj.field_f64("stars_per_min", self.stars_per_min);
        obj.field_f64("steps_per_sec", self.steps_per_sec);
        obj.field_f64("zone_steps_per_sec", self.zone_steps_per_sec);
        obj.field_f64("speedup_vs_n1", self.speedup_vs_n1);
        obj.field_usize("current_rss_kib", self.current_rss_kib);
        obj.field_usize("peak_rss_kib", self.peak_rss_kib);
        obj.finish()
    }
}

fn load_template_model_from_case(
    orig_case_dir: &Path,
) -> Result<StarModel, String> {
    let staging = StagedCaseGuard::new(orig_case_dir)?;
    let case_dir = staging.staged_dir.as_path();
    let rn_stages = parse_rn_stages(case_dir, true)
        .map_err(|e| format!("parse_rn_stages failed: {e}"))?;
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls: true,
    };
    let mut extras = DefaultStarExtras;
    let mut current_model: Option<StarModel> = None;

    for stage in &rn_stages {
        match stage {
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                current_model = None;
                fs::copy(case_dir.join(src_mod), case_dir.join(dst_mod))
                    .map_err(|e| format!("copy checkpoint failed: {e}"))?;
            }
            RnStage::DoOne { inlist, .. } => {
                let bundle = InlistBundle::from_file_in_case_dir_with_options(
                    case_dir, inlist, load_opts,
                )
                .map_err(|e| format!("inlist load failed: {e}"))?;
                let model = init_or_update_model_from_bundle(
                    &bundle,
                    current_model.take(),
                    &mut extras,
                )
                .map_err(|e| format!("model init failed: {e}"))?;
                return Ok(model);
            }
        }
    }
    Err(String::from("no DoOne stage found in case rn"))
}

fn evolve_one_grid_star(
    model: &mut StarModel,
    max_steps: Option<usize>,
) -> Result<(usize, usize), String> {
    let mut extras = DefaultStarExtras;
    let do_remesh =
        model.controls.okay_to_remesh && model.controls.mesh_delta_coeff > 0.0;
    let op_split = model.controls.op_split_burn;
    let cap = max_steps.unwrap_or(50_000);
    let mut steps = 0_usize;
    let mut zone_steps = 0_usize;

    for _ in 0..cap {
        star_evolve_step_with_extras(
            model,
            do_remesh,
            op_split,
            true,
            &mut extras,
        )
        .map_err(|e| format!("grid star step failed: {e}"))?;
        steps += 1;
        zone_steps += model.nz;
        model.mstar_start = model.mstar;
        if !model.controls.rsp_flag {
            model
                .sync_mesh_and_pack()
                .map_err(|e| format!("sync_mesh failed: {e}"))?;
            set_hydro_vars(model, false, false)
                .map_err(|e| format!("set_hydro_vars failed: {e}"))?;
            model
                .save_start_values()
                .map_err(|e| format!("save_start_values failed: {e}"))?;
        }
        let hr = extract_history_row(model);
        if matches!(
            check_termination_limits_with_history(model, Some(&hr)),
            Ok(flexmesa_star::LimitCheckOutcome::Terminate { .. })
        ) {
            break;
        }
    }
    Ok((steps, zone_steps))
}

/// Runs `batch_sizes` independent stellar evolutions concurrently on a dedicated
/// `pool_threads` `rayon::ThreadPool` and serializes the scaling curve as JSON.
pub fn run_batch_scaling_json(
    orig_case_dir: &Path,
    batch_sizes: &[usize],
    pool_threads: usize,
    max_steps: Option<usize>,
    heterogeneous_alpha: bool,
) -> Result<String, String> {
    let warmup_sec = warmup_static_tables()?;
    let case_name = orig_case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let base_model = load_template_model_from_case(orig_case_dir)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads.max(1))
        .build()
        .map_err(|e| format!("failed to build rayon ThreadPool: {e}"))?;

    let mut records = Vec::with_capacity(batch_sizes.len());
    let mut n1_stars_per_min = 0.0_f64;

    for &n_stars in batch_sizes {
        let count = n_stars.max(1);
        let mut models: Vec<StarModel> = (0..count)
            .map(|idx| {
                let mut m = base_model.clone();
                if heterogeneous_alpha && count > 1 {
                    let delta =
                        ((idx as f64) / ((count - 1) as f64) - 0.5) * 0.04;
                    m.controls.mixing_length_alpha += delta;
                }
                m
            })
            .collect();

        let t0 = Instant::now();
        let outcomes: Result<Vec<(usize, usize)>, String> =
            pool.install(|| {
                models
                    .par_iter_mut()
                    .map(|m| evolve_one_grid_star(m, max_steps))
                    .collect()
            });
        let wall_sec = t0.elapsed().as_secs_f64().max(1.0e-9);
        let pairs = outcomes?;
        let total_steps: usize = pairs.iter().map(|p| p.0).sum();
        let total_zone_steps: usize = pairs.iter().map(|p| p.1).sum();

        let stars_per_min = 60.0 * (count as f64) / wall_sec;
        if n1_stars_per_min <= 0.0 {
            n1_stars_per_min = stars_per_min;
        }
        let speedup_vs_n1 = stars_per_min / n1_stars_per_min.max(1.0e-12);

        records.push(BatchPointRecord {
            n_stars: count,
            pool_threads: pool_threads.max(1),
            wall_sec,
            total_steps,
            total_zone_steps,
            stars_per_min,
            steps_per_sec: (total_steps as f64) / wall_sec,
            zone_steps_per_sec: (total_zone_steps as f64) / wall_sec,
            speedup_vs_n1,
            current_rss_kib: current_rss_kib(),
            peak_rss_kib: peak_rss_kib(),
        });
    }

    let mut arr = String::from("[\n");
    for (idx, rec) in records.iter().enumerate() {
        if idx > 0 {
            arr.push_str(",\n");
        }
        arr.push_str(&rec.to_json(4));
    }
    arr.push_str("\n  ]");

    let mut root = JsonObjectBuilder::new(0);
    root.field_str("case", &case_name);
    root.field_str("numerics_mode", numerics_mode().as_str());
    root.field_usize("pool_threads", pool_threads.max(1));
    root.field_usize("max_steps_cap", max_steps.unwrap_or(0));
    root.field_bool("heterogeneous_alpha", heterogeneous_alpha);
    root.field_f64("warmup_wall_sec", warmup_sec);
    root.field_usize("template_zones", base_model.nz);
    root.field_usize("template_species", base_model.species);
    root.field_usize("star_model_heap_bytes", base_model.heap_bytes());
    root.field_raw("points", &arr);
    Ok(root.finish())
}
