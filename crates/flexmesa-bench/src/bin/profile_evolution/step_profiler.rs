// SPDX-License-Identifier: LGPL-3.0-or-later
//! Real-evolution trajectory timer, per-step phase decomposition, and `#150`
//! waste-removal profiler (`#168`).

use std::collections::BTreeSet;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use flexmesa_math::{exp10, math_init, numerics_mode};
use flexmesa_mtx::{solve_block_tridiag_bcyclic, solve_block_tridiagonal};
use flexmesa_star::controls::{StarControls, StarError, StarJobControls};
use flexmesa_star::eps_grav::eval_eps_grav;
use flexmesa_star::history::{extract_history_row, HistoryRow};
use flexmesa_star::hydro_eqns::{assemble_henyey_system, HenyeySystem};
use flexmesa_star::hydro_vars::{
    do_brunt_b, do_brunt_n2, eval_surf_bc, set_hydro_vars, set_mlt_vars,
};
use flexmesa_star::inlist::{InlistBundle, InlistLoadOptions};
use flexmesa_star::inlist_runner::{
    run_inlist_stage_with_logs, ExtrasDirective, StarExtras, StarLogsOptions,
};
use flexmesa_star::mesh_adjust::{
    plan_mesh, remap_onto_dq, MeshFunctionConfig,
};
use flexmesa_star::micro::{
    do_eos, do_kap, do_net, do_neu, resolve_model_net, set_basic_vars,
};
use flexmesa_star::model::StarModel;
use flexmesa_star::model_io::{
    compute_photo_checksum, write_mod_string, write_photo_bytes,
};
use flexmesa_star::primary_state::PrimaryStateBuffer;
use flexmesa_star::rn_parser::{parse_rn_stages, RnStage};
use flexmesa_star::solver::{
    compute_corr_coeff, compute_model_correction_norms, compute_residual_norms,
};
use flexmesa_star::star_job_init::init_or_update_model_from_bundle;
use flexmesa_star::struct_burn_mix::op_split_burn_zones;
use flexmesa_star::timestep_limits::timestep_controller;
use flexmesa_star::zams::load_zams_model;
use flexmesa_star::{
    eval_total_energy_integrals, update_step_energy_conservation,
};

use crate::json_fmt::{
    current_rss_kib, peak_rss_kib, JsonObjectBuilder, StagedCaseGuard,
};

/// Warms up `flexmesa-math`, `HelmTable`, `KapTables`, and `basic.net` static
/// tables so cold-start disk `mmap` and spline loads are isolated from
/// steady-state per-step timings.
pub fn warmup_static_tables() -> Result<f64, String> {
    let t0 = Instant::now();
    math_init();
    let ctrl = StarControls {
        zbase: 0.02,
        ..StarControls::default()
    };
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .map_err(|e| format!("warmup load_zams_model failed: {e}"))?;
    set_hydro_vars(&mut model, false, false)
        .map_err(|e| format!("warmup set_hydro_vars failed: {e}"))?;
    Ok(t0.elapsed().as_secs_f64())
}

/// Accumulated phase timings and call counts across a full evolution.
#[derive(Debug, Clone, Default)]
pub struct PhaseBreakdown {
    pub eos_sec: f64,
    pub kap_sec: f64,
    pub neu_sec: f64,
    pub net_sec: f64,
    pub net_and_jacobian_sec: f64,
    pub turbulence_sec: f64,
    pub equation_assembly_sec: f64,
    pub block_solve_sec: f64,
    pub remesh_sec: f64,
    pub direct_bookkeeping_measured_sec: f64,
    pub bookkeeping_and_copies_sec: f64,
    pub bottom_up_total_sec: f64,
    pub reference_evolve_wall_sec: f64,
    pub eos_eval_calls: usize,
    pub kap_eval_calls: usize,
    pub neu_eval_calls: usize,
    pub net_jac_eval_calls: usize,
    pub mlt_eval_calls: usize,
    pub assembly_calls: usize,
    pub linear_solve_calls: usize,
    pub remesh_plan_calls: usize,
    pub remesh_applied_calls: usize,
    pub saved_micro_eval_sec: f64,
    pub pre150_full_clone_every_step_sec: f64,
    pub post150_lazy_primary_buf_sec: f64,
}

impl PhaseBreakdown {
    /// Serializes this phase breakdown as a JSON object string.
    #[must_use]
    pub fn to_json(&self, steps: usize, indent: usize) -> String {
        let denom = self.reference_evolve_wall_sec.max(1.0e-12);
        let steps_f = (steps.max(1)) as f64;
        let mut obj = JsonObjectBuilder::new(indent);
        obj.field_f64(
            "reference_evolve_wall_sec",
            self.reference_evolve_wall_sec,
        );
        obj.field_f64("bottom_up_total_sec", self.bottom_up_total_sec);
        obj.field_f64("eos_sec", self.eos_sec);
        obj.field_f64("eos_share_pct", 100.0 * self.eos_sec / denom);
        obj.field_f64("kap_sec", self.kap_sec);
        obj.field_f64("kap_share_pct", 100.0 * self.kap_sec / denom);
        obj.field_f64("neu_sec", self.neu_sec);
        obj.field_f64("net_sec", self.net_sec);
        obj.field_f64("net_and_jacobian_sec", self.net_and_jacobian_sec);
        obj.field_f64(
            "net_and_jacobian_share_pct",
            100.0 * self.net_and_jacobian_sec / denom,
        );
        obj.field_f64("turbulence_sec", self.turbulence_sec);
        obj.field_f64(
            "turbulence_share_pct",
            100.0 * self.turbulence_sec / denom,
        );
        obj.field_f64("equation_assembly_sec", self.equation_assembly_sec);
        obj.field_f64(
            "equation_assembly_share_pct",
            100.0 * self.equation_assembly_sec / denom,
        );
        obj.field_f64("block_solve_sec", self.block_solve_sec);
        obj.field_f64(
            "block_solve_share_pct",
            100.0 * self.block_solve_sec / denom,
        );
        obj.field_f64("remesh_sec", self.remesh_sec);
        obj.field_f64("remesh_share_pct", 100.0 * self.remesh_sec / denom);
        obj.field_f64(
            "direct_bookkeeping_measured_sec",
            self.direct_bookkeeping_measured_sec,
        );
        obj.field_f64(
            "bookkeeping_and_copies_sec",
            self.bookkeeping_and_copies_sec,
        );
        obj.field_f64(
            "bookkeeping_and_copies_share_pct",
            100.0 * self.bookkeeping_and_copies_sec / denom,
        );
        obj.field_usize("eos_eval_calls", self.eos_eval_calls);
        obj.field_usize("kap_eval_calls", self.kap_eval_calls);
        obj.field_usize("neu_eval_calls", self.neu_eval_calls);
        obj.field_usize("net_jac_eval_calls", self.net_jac_eval_calls);
        obj.field_usize("mlt_eval_calls", self.mlt_eval_calls);
        obj.field_usize("assembly_calls", self.assembly_calls);
        obj.field_usize("linear_solve_calls", self.linear_solve_calls);
        obj.field_usize("remesh_plan_calls", self.remesh_plan_calls);
        obj.field_usize("remesh_applied_calls", self.remesh_applied_calls);
        obj.field_f64(
            "eos_evals_per_accepted_step",
            (self.eos_eval_calls as f64) / steps_f,
        );
        obj.field_f64(
            "kap_evals_per_accepted_step",
            (self.kap_eval_calls as f64) / steps_f,
        );
        obj.field_f64(
            "neu_evals_per_accepted_step",
            (self.neu_eval_calls as f64) / steps_f,
        );
        obj.field_f64(
            "net_jac_evals_per_accepted_step",
            (self.net_jac_eval_calls as f64) / steps_f,
        );
        obj.field_f64(
            "mlt_evals_per_accepted_step",
            (self.mlt_eval_calls as f64) / steps_f,
        );
        obj.field_f64(
            "assembly_calls_per_accepted_step",
            (self.assembly_calls as f64) / steps_f,
        );
        obj.field_f64(
            "linear_solves_per_accepted_step",
            (self.linear_solve_calls as f64) / steps_f,
        );
        obj.field_f64("saved_micro_eval_sec", self.saved_micro_eval_sec);
        obj.field_f64(
            "pre150_full_clone_every_step_sec",
            self.pre150_full_clone_every_step_sec,
        );
        obj.field_f64(
            "post150_lazy_primary_buf_sec",
            self.post150_lazy_primary_buf_sec,
        );
        obj.finish()
    }
}

/// Complete telemetry record for a single `star/test_suite/<case>` run.
#[derive(Debug, Clone)]
pub struct CaseProfileRecord {
    pub case_name: String,
    pub numerics_mode: String,
    pub rayon_threads: usize,
    pub allow_unsupported_controls: bool,
    pub n_controls_ignored: usize,
    pub ignored_controls: Vec<String>,
    pub run_mode: String,
    pub net_name: String,
    pub termination_code: String,
    pub warmup_wall_sec: f64,
    pub init_wall_sec: f64,
    pub evolve_wall_sec: f64,
    pub total_wall_sec: f64,
    pub steps_taken: usize,
    pub total_zone_steps: usize,
    pub total_zone_newton_iters: usize,
    pub initial_zones: usize,
    pub final_zones: usize,
    pub species: usize,
    pub nvar_coupled: usize,
    pub total_newton_iterations: usize,
    pub accepted_step_newton_iterations: usize,
    pub total_solver_calls: usize,
    pub total_retries: usize,
    pub total_redos: usize,
    pub remesh_applied_steps: usize,
    pub micro_cache_hits: usize,
    pub micro_cache_misses: usize,
    pub photo_checksum: u64,
    pub star_model_heap_bytes: usize,
    pub primary_state_buffer_heap_bytes: usize,
    pub current_rss_kib: usize,
    pub peak_rss_kib: usize,
    pub final_history: HistoryRow,
    pub phases: Option<PhaseBreakdown>,
}

impl CaseProfileRecord {
    /// Serializes this record as a formatted JSON object.
    #[must_use]
    pub fn to_json(&self) -> String {
        let steps_f = (self.steps_taken.max(1)) as f64;
        let zs_f = (self.total_zone_steps.max(1)) as f64;
        let nit_f = (self.total_newton_iterations.max(1)) as f64;
        let znit_f = (self.total_zone_newton_iters.max(1)) as f64;
        let mut obj = JsonObjectBuilder::new(0);
        obj.field_str("case", &self.case_name);
        obj.field_str("numerics_mode", &self.numerics_mode);
        obj.field_usize("rayon_threads", self.rayon_threads);
        obj.field_bool(
            "allow_unsupported_controls",
            self.allow_unsupported_controls,
        );
        obj.field_usize("n_controls_ignored", self.n_controls_ignored);
        obj.field_str_array("ignored_controls", &self.ignored_controls);
        obj.field_str("run_mode", &self.run_mode);
        obj.field_str("net_name", &self.net_name);
        obj.field_str("termination_code", &self.termination_code);
        obj.field_f64("warmup_wall_sec", self.warmup_wall_sec);
        obj.field_f64("init_wall_sec", self.init_wall_sec);
        obj.field_f64("evolve_wall_sec", self.evolve_wall_sec);
        obj.field_f64("total_wall_sec", self.total_wall_sec);
        obj.field_f64(
            "wall_ms_per_step",
            1000.0 * self.evolve_wall_sec / steps_f,
        );
        obj.field_f64(
            "wall_us_per_zone_step",
            1.0e6 * self.evolve_wall_sec / zs_f,
        );
        obj.field_f64(
            "wall_ms_per_newton_iter",
            1000.0 * self.evolve_wall_sec / nit_f,
        );
        obj.field_f64(
            "wall_us_per_zone_newton_iter",
            1.0e6 * self.evolve_wall_sec / znit_f,
        );
        obj.field_usize("steps_taken", self.steps_taken);
        obj.field_usize("total_zone_steps", self.total_zone_steps);
        obj.field_usize(
            "total_zone_newton_iters",
            self.total_zone_newton_iters,
        );
        obj.field_usize("initial_zones", self.initial_zones);
        obj.field_usize("final_zones", self.final_zones);
        obj.field_f64("mean_zones", (self.total_zone_steps as f64) / steps_f);
        obj.field_usize("species", self.species);
        obj.field_usize("nvar_coupled", self.nvar_coupled);
        obj.field_usize(
            "total_newton_iterations",
            self.total_newton_iterations,
        );
        obj.field_usize(
            "accepted_step_newton_iterations",
            self.accepted_step_newton_iterations,
        );
        obj.field_f64(
            "newton_iters_per_step",
            (self.total_newton_iterations as f64) / steps_f,
        );
        obj.field_usize("total_solver_calls", self.total_solver_calls);
        obj.field_usize("total_retries", self.total_retries);
        obj.field_usize("total_redos", self.total_redos);
        obj.field_usize("remesh_applied_steps", self.remesh_applied_steps);
        obj.field_usize("micro_cache_hits", self.micro_cache_hits);
        obj.field_usize("micro_cache_misses", self.micro_cache_misses);
        obj.field_f64(
            "micro_cache_hits_per_step",
            (self.micro_cache_hits as f64) / steps_f,
        );
        obj.field_f64(
            "micro_cache_misses_per_step",
            (self.micro_cache_misses as f64) / steps_f,
        );
        obj.field_hex_u64("photo_checksum", self.photo_checksum);
        obj.field_usize("star_model_heap_bytes", self.star_model_heap_bytes);
        obj.field_usize(
            "primary_state_buffer_heap_bytes",
            self.primary_state_buffer_heap_bytes,
        );
        obj.field_usize("current_rss_kib", self.current_rss_kib);
        obj.field_usize("peak_rss_kib", self.peak_rss_kib);
        obj.field_f64("final_star_age_yr", self.final_history.star_age);
        obj.field_f64("final_log_l", self.final_history.log_l);
        obj.field_f64("final_log_teff", self.final_history.log_teff);
        obj.field_f64("final_teff_k", exp10(self.final_history.log_teff));
        obj.field_f64("final_log_r", self.final_history.log_r);
        obj.field_f64("final_log_center_t", self.final_history.log_center_t);
        obj.field_f64(
            "final_log_center_rho",
            self.final_history.log_center_rho,
        );
        obj.field_f64("final_center_h1", self.final_history.center_h1);
        obj.field_f64("final_center_he4", self.final_history.center_he4);
        if let Some(ref ph) = self.phases {
            obj.field_raw("phases", &ph.to_json(self.steps_taken, 2));
        }
        obj.finish()
    }
}

struct StepKernelSample {
    eos_sec: f64,
    kap_sec: f64,
    neu_sec: f64,
    net_jac_sec: f64,
    turb_sec: f64,
    cache_hit_copy_sec: f64,
    assembly_sec: f64,
    block_solve_sec: f64,
    remesh_plan_sec: f64,
    remesh_remap_sec: f64,
    star_clone_sec: f64,
    primary_buf_sec: f64,
    step_bookkeeping_sec: f64,
}

fn measure_step_kernels_on_probe(
    model: &StarModel,
    do_remesh: bool,
    did_remap: bool,
) -> Result<StepKernelSample, StarError> {
    let t_clone0 = Instant::now();
    let mut probe = black_box(model.clone());
    let star_clone_sec = t_clone0.elapsed().as_secs_f64();

    let t_pbuf0 = Instant::now();
    let pbuf = black_box(PrimaryStateBuffer::from_model(&probe));
    let primary_buf_sec = t_pbuf0.elapsed().as_secs_f64();
    black_box(&pbuf);

    let net_info = resolve_model_net(&probe)?;
    let do_burn =
        probe.controls.do_burn && probe.controls.dxdt_nuc_factor > 0.0;
    let do_mix = probe.controls.do_mix && probe.controls.mix_factor > 0.0;
    let include_chem = (do_burn || do_mix) && probe.species > 0;

    let t0 = Instant::now();
    set_basic_vars(&mut probe)?;
    do_eos(&mut probe)?;
    let eos_sec = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    do_kap(&mut probe)?;
    let kap_sec = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    do_neu(&mut probe)?;
    let neu_sec = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    do_net(&mut probe, &net_info, false)?;
    let dt_probe = probe.dt;
    if probe.controls.op_split_burn && do_burn && dt_probe > 0.0 {
        op_split_burn_zones(&mut probe, dt_probe)?;
    }
    let net_jac_sec = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    do_brunt_b(&mut probe)?;
    set_mlt_vars(&mut probe)?;
    do_brunt_n2(&mut probe);
    eval_surf_bc(&mut probe)?;
    eval_eps_grav(&mut probe);
    let turb_sec = t0.elapsed().as_secs_f64();

    // Prime `MicrophysicsCache` on `probe`, then time a warm `set_hydro_vars` hit.
    probe.aux.micro_cache.clear();
    set_hydro_vars(&mut probe, false, false)?;
    let t0 = Instant::now();
    set_hydro_vars(&mut probe, false, false)?;
    let warm_hydro_sec = t0.elapsed().as_secs_f64();
    let cache_hit_copy_sec = (warm_hydro_sec - turb_sec).max(0.0);

    probe.save_start_values()?;
    probe.pack_xh();
    if include_chem && probe.species > 0 {
        probe.xa_sub_xa_start.resize(probe.xa.len(), 0.0);
        probe.xa_sub_xa_start.fill(0.0);
    }
    let mut sys = HenyeySystem::new(0, 0, Vec::new());
    let t0 = Instant::now();
    assemble_henyey_system(&mut probe, include_chem, &mut sys)?;
    let assembly_sec = t0.elapsed().as_secs_f64();

    let rhs: Vec<f64> = sys.equ.iter().map(|&r| -r).collect();
    let t0 = Instant::now();
    let dx_scaled = if sys.nz >= 2 {
        solve_block_tridiag_bcyclic(
            sys.nvar, sys.nz, &sys.lblk, &sys.dblk, &sys.ublk, &rhs, true,
        )
    } else {
        solve_block_tridiagonal(
            sys.nvar, sys.nz, &sys.lblk, &sys.dblk, &sys.ublk, &rhs,
        )
    }
    .map_err(|e| StarError::ConvergenceFailure(format!("probe solve: {e}")))?;
    let _ = compute_model_correction_norms(
        &probe,
        sys.nz,
        sys.nvar,
        &dx_scaled,
        &sys.x_scale,
    );
    let _ =
        compute_corr_coeff(&probe, sys.nz, sys.nvar, &dx_scaled, &sys.x_scale);
    let _ = compute_residual_norms(&probe, sys.nvar, &sys.equ);
    let block_solve_sec = t0.elapsed().as_secs_f64();

    let (mut remesh_plan_sec, mut remesh_remap_sec) = (0.0, 0.0);
    if do_remesh {
        let t0 = Instant::now();
        let mesh_cfg = MeshFunctionConfig::from_controls(&probe.controls);
        let (dq_new, ns, nm) = plan_mesh(&probe, &mesh_cfg);
        remesh_plan_sec = t0.elapsed().as_secs_f64();
        if did_remap && (ns + nm > 0) {
            let t1 = Instant::now();
            remap_onto_dq(&mut probe, &dq_new)?;
            remesh_remap_sec = t1.elapsed().as_secs_f64();
        }
    }

    let t0 = Instant::now();
    probe.save_old_values();
    probe.save_start_values()?;
    let xh_backup = black_box(probe.xh.clone());
    let species_backup = black_box(probe.xa.clone());
    black_box((xh_backup, species_backup));
    let _ = black_box(eval_total_energy_integrals(&probe));
    let dt_econs = probe.dt.max(1.0);
    update_step_energy_conservation(&mut probe, dt_econs);
    let _ = black_box(timestep_controller(&mut probe));
    let _ = black_box(extract_history_row(&probe));
    let step_bookkeeping_sec = t0.elapsed().as_secs_f64();

    Ok(StepKernelSample {
        eos_sec,
        kap_sec,
        neu_sec,
        net_jac_sec,
        turb_sec,
        cache_hit_copy_sec,
        assembly_sec,
        block_solve_sec,
        remesh_plan_sec,
        remesh_remap_sec,
        star_clone_sec,
        primary_buf_sec,
        step_bookkeeping_sec,
    })
}

#[derive(Default)]
struct ProfileExtras {
    with_phase_profile: bool,
    do_remesh: bool,
    total_zone_steps: usize,
    total_zone_newton_iters: usize,
    remesh_applied_steps: usize,
    prev_hits: usize,
    prev_misses: usize,
    prev_solver_iters: usize,
    prev_retries: usize,
    prev_nz: usize,
    pre_state_for_probe: Option<StarModel>,
    last_probe_sample: Option<StepKernelSample>,
    phases: PhaseBreakdown,
    probe_error: Option<String>,
}

impl ProfileExtras {
    fn reset_stage(&mut self, model: &StarModel) {
        self.do_remesh = model.controls.okay_to_remesh
            && model.controls.mesh_delta_coeff > 0.0;
        self.prev_hits = model.micro_cache_hits;
        self.prev_misses = model.micro_cache_misses;
        self.prev_solver_iters = model.total_num_solver_iterations;
        self.prev_retries = model.num_retries;
        self.prev_nz = model.nz;
        self.pre_state_for_probe = None;
        self.last_probe_sample = None;
    }

    fn record_post_step_hydro(
        &mut self,
        d_hits: usize,
        d_misses: usize,
        sample: &StepKernelSample,
    ) {
        let miss_f = d_misses as f64;
        let hit_f = d_hits as f64;
        let hydro_calls = d_misses + d_hits;
        self.phases.eos_sec += miss_f * sample.eos_sec;
        self.phases.kap_sec += miss_f * sample.kap_sec;
        self.phases.neu_sec += miss_f * sample.neu_sec;
        self.phases.net_sec += miss_f * sample.net_jac_sec;
        self.phases.turbulence_sec += (hydro_calls as f64) * sample.turb_sec;
        let raw_micro = sample.eos_sec
            + sample.kap_sec
            + sample.neu_sec
            + sample.net_jac_sec;
        self.phases.saved_micro_eval_sec +=
            hit_f * (raw_micro - sample.cache_hit_copy_sec).max(0.0);
        self.phases.direct_bookkeeping_measured_sec +=
            hit_f * sample.cache_hit_copy_sec;
        self.phases.eos_eval_calls += d_misses;
        self.phases.kap_eval_calls += d_misses;
        self.phases.neu_eval_calls += d_misses;
        self.phases.net_jac_eval_calls += d_misses;
        self.phases.mlt_eval_calls += hydro_calls;
    }
}

impl StarExtras for ProfileExtras {
    fn extras_start_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        if self.with_phase_profile {
            self.pre_state_for_probe = Some(model.clone());
        }
        ExtrasDirective::KeepGoing
    }

    fn extras_finish_step(&mut self, model: &mut StarModel) -> ExtrasDirective {
        let d_solver_iters = model
            .total_num_solver_iterations
            .saturating_sub(self.prev_solver_iters);
        let d_retries = model.num_retries.saturating_sub(self.prev_retries);
        self.prev_solver_iters = model.total_num_solver_iterations;
        self.prev_retries = model.num_retries;
        self.total_zone_steps += model.nz;
        self.total_zone_newton_iters += model.nz * d_solver_iters;

        if self.with_phase_profile {
            let d_hits = model.micro_cache_hits.saturating_sub(self.prev_hits);
            let d_misses =
                model.micro_cache_misses.saturating_sub(self.prev_misses);
            self.prev_hits = model.micro_cache_hits;
            self.prev_misses = model.micro_cache_misses;
            if let Some(pre_m) = self.pre_state_for_probe.take() {
                let did_remap = pre_m.nz != model.nz || pre_m.dq != model.dq;
                if did_remap {
                    self.remesh_applied_steps += 1;
                }
                match measure_step_kernels_on_probe(
                    &pre_m,
                    self.do_remesh,
                    did_remap,
                ) {
                    Ok(sample) => {
                        let n_asm = 1 + d_solver_iters + d_retries;
                        let n_lin = d_solver_iters;
                        self.record_post_step_hydro(d_hits, d_misses, &sample);
                        self.phases.equation_assembly_sec +=
                            (n_asm as f64) * sample.assembly_sec;
                        self.phases.block_solve_sec +=
                            (n_lin as f64) * sample.block_solve_sec;
                        if self.do_remesh {
                            self.phases.remesh_plan_calls += 1;
                            self.phases.remesh_sec += sample.remesh_plan_sec
                                + sample.remesh_remap_sec;
                            self.phases.pre150_full_clone_every_step_sec +=
                                sample.star_clone_sec;
                        }
                        if did_remap {
                            self.phases.remesh_applied_calls += 1;
                            self.phases.post150_lazy_primary_buf_sec +=
                                sample.primary_buf_sec;
                        }
                        self.phases.direct_bookkeeping_measured_sec += sample
                            .step_bookkeeping_sec
                            + if did_remap {
                                sample.primary_buf_sec
                            } else {
                                0.0
                            };
                        self.phases.assembly_calls += n_asm;
                        self.phases.linear_solve_calls += n_lin;
                        self.last_probe_sample = Some(sample);
                    }
                    Err(err) => {
                        self.probe_error = Some(format!("phase probe: {err}"));
                        return ExtrasDirective::Error(format!(
                            "phase probe failed: {err}"
                        ));
                    }
                }
            }
        } else if model.nz != self.prev_nz {
            self.remesh_applied_steps += 1;
        }
        self.prev_nz = model.nz;
        ExtrasDirective::KeepGoing
    }

    fn extras_after_evolve(
        &mut self,
        model: &mut StarModel,
    ) -> Result<(), StarError> {
        if let Some(ref err) = self.probe_error {
            return Err(StarError::ConvergenceFailure(err.clone()));
        }
        if self.with_phase_profile {
            let d_hits = model.micro_cache_hits.saturating_sub(self.prev_hits);
            let d_misses =
                model.micro_cache_misses.saturating_sub(self.prev_misses);
            self.prev_hits = model.micro_cache_hits;
            self.prev_misses = model.micro_cache_misses;
            if d_hits > 0 || d_misses > 0 {
                if let Some(sample) = self.last_probe_sample.take() {
                    self.record_post_step_hydro(d_hits, d_misses, &sample);
                    self.last_probe_sample = Some(sample);
                }
            }
        }
        Ok(())
    }
}

/// Runs a staged `star/test_suite/<case>` directory cleanly through the
/// canonical `run_inlist_stage_with_logs` driver (with optional per-step
/// kernel phase decomposition and optional full-history or `LOGS/` output).
pub fn profile_star_case(
    orig_case_dir: &Path,
    allow_unsupported_controls: bool,
    max_steps_per_stage: Option<usize>,
    with_phase_profile: bool,
    reference_evolve_sec_override: Option<f64>,
    history_out: Option<&Path>,
    logs_dir_out: Option<&Path>,
) -> Result<CaseProfileRecord, String> {
    let warmup_wall_sec = warmup_static_tables()?;
    let mode_str = numerics_mode().as_str().to_string();
    let rayon_threads = rayon::current_num_threads();

    let t_total0 = Instant::now();
    let staging = StagedCaseGuard::new(orig_case_dir)?;
    let case_dir = staging.staged_dir.as_path();
    let case_name = orig_case_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let rn_stages = parse_rn_stages(case_dir, true)
        .map_err(|e| format!("parse_rn_stages failed: {e}"))?;
    let load_opts = InlistLoadOptions {
        allow_unsupported_controls,
    };
    let mut extras = ProfileExtras {
        with_phase_profile,
        ..ProfileExtras::default()
    };
    let mut current_model: Option<StarModel> = None;

    let mut init_wall_sec = 0.0_f64;
    let mut evolve_wall_sec = 0.0_f64;
    let mut steps_taken = 0_usize;
    let mut initial_zones = 0_usize;
    let mut total_newton_iterations = 0_usize;
    let mut accepted_step_newton_iterations = 0_usize;
    let mut total_solver_calls = 0_usize;
    let mut total_retries = 0_usize;
    let mut total_redos = 0_usize;
    let mut micro_cache_hits = 0_usize;
    let mut micro_cache_misses = 0_usize;
    let mut termination_code = String::from("no_stages_executed");
    let mut ignored_set: BTreeSet<String> = BTreeSet::new();

    for stage in &rn_stages {
        match stage {
            RnStage::CopyCheckpoint { src_mod, dst_mod } => {
                current_model = None;
                let (src, dst) =
                    (case_dir.join(src_mod), case_dir.join(dst_mod));
                fs::copy(&src, &dst).map_err(|e| {
                    format!(
                        "checkpoint copy {src_mod} -> {dst_mod} failed: {e}"
                    )
                })?;
            }
            RnStage::DoOne {
                inlist,
                output_mod,
                logs_dir,
            } => {
                let t_init0 = Instant::now();
                let bundle = InlistBundle::from_file_in_case_dir_with_options(
                    case_dir, inlist, load_opts,
                )
                .map_err(|e| format!("inlist load failed: {e}"))?;
                for sec in &bundle.validation_report.unsupported_sections {
                    ignored_set.insert(format!("&{sec}"));
                }
                for (sec, key) in &bundle.validation_report.unsupported_controls
                {
                    ignored_set.insert(format!("&{sec}::{key}"));
                }
                let mut model = init_or_update_model_from_bundle(
                    &bundle,
                    current_model.take(),
                    &mut extras,
                )
                .map_err(|e| format!("model init failed: {e}"))?;
                init_wall_sec += t_init0.elapsed().as_secs_f64();

                if initial_zones == 0 {
                    initial_zones = model.nz;
                }
                let base_hits = model.micro_cache_hits;
                let base_misses = model.micro_cache_misses;
                let base_solver_iters = model.total_num_solver_iterations;
                let base_solver_calls = model.total_num_solver_calls;
                let base_redos = model.num_redos;
                extras.reset_stage(&model);

                let mut logs_writer = if let Some(l_dir) = logs_dir_out {
                    let logs_opts =
                        StarLogsOptions::from_target(l_dir, true, false, true);
                    logs_opts
                        .build_writer(&model, case_dir, logs_dir.as_deref())
                        .map_err(|e| format!("logs writer init failed: {e}"))?
                } else if let Some(h_path) = history_out {
                    let logs_opts = StarLogsOptions::from_target(
                        h_path, false, false, false,
                    );
                    logs_opts
                        .build_writer(&model, case_dir, logs_dir.as_deref())
                        .map_err(|e| format!("logs writer init failed: {e}"))?
                } else {
                    None
                };

                let stage_rep = run_inlist_stage_with_logs(
                    &mut model,
                    &bundle,
                    &mut extras,
                    max_steps_per_stage,
                    logs_writer.as_mut(),
                )
                .map_err(|e| format!("stage evolution failed: {e}"))?;

                evolve_wall_sec += stage_rep.runtime_sec;
                steps_taken += stage_rep.steps_taken;
                accepted_step_newton_iterations +=
                    stage_rep.total_newton_iterations;
                total_retries += stage_rep.total_retries;
                termination_code = stage_rep.termination_code;

                total_newton_iterations += model
                    .total_num_solver_iterations
                    .saturating_sub(base_solver_iters);
                micro_cache_hits +=
                    model.micro_cache_hits.saturating_sub(base_hits);
                micro_cache_misses +=
                    model.micro_cache_misses.saturating_sub(base_misses);
                total_solver_calls += model
                    .total_num_solver_calls
                    .saturating_sub(base_solver_calls);
                total_redos += model.num_redos.saturating_sub(base_redos);

                if let Some(out_name) = output_mod {
                    let out_path = case_dir.join(out_name);
                    if !out_path.exists() {
                        let txt = write_mod_string(&model)
                            .map_err(|e| format!("write_mod failed: {e}"))?;
                        fs::write(&out_path, txt).map_err(|e| {
                            format!("fs::write mod failed: {e}")
                        })?;
                    }
                }
                current_model = Some(model);
            }
        }
    }

    let total_wall_sec = t_total0.elapsed().as_secs_f64();
    let final_model = current_model
        .ok_or_else(|| String::from("no model produced across case stages"))?;
    let final_history = extract_history_row(&final_model);
    let photo_bytes = write_photo_bytes(&final_model);
    let photo_checksum = compute_photo_checksum(&photo_bytes);
    let star_model_heap_bytes = final_model.heap_bytes();
    let pbuf_final = PrimaryStateBuffer::from_model(&final_model);
    let primary_state_buffer_heap_bytes = pbuf_final.heap_bytes();
    let do_burn = final_model.controls.do_burn
        && final_model.controls.dxdt_nuc_factor > 0.0;
    let do_mix =
        final_model.controls.do_mix && final_model.controls.mix_factor > 0.0;
    let include_chem = (do_burn || do_mix) && final_model.species > 0;
    let nvar_coupled = final_model.layout.nvar(include_chem);

    let ignored_controls: Vec<String> = ignored_set.into_iter().collect();
    let n_controls_ignored = ignored_controls.len();
    let run_mode = if !allow_unsupported_controls || n_controls_ignored == 0 {
        String::from("strict")
    } else {
        format!("diagnostic ({n_controls_ignored} controls ignored)")
    };

    let mut phases = extras.phases;
    let phases_opt = if with_phase_profile {
        phases.net_and_jacobian_sec = phases.neu_sec + phases.net_sec;
        let ref_sec = reference_evolve_sec_override.unwrap_or(evolve_wall_sec);
        let non_bookkeeping = phases.eos_sec
            + phases.kap_sec
            + phases.net_and_jacobian_sec
            + phases.turbulence_sec
            + phases.equation_assembly_sec
            + phases.block_solve_sec
            + phases.remesh_sec;
        phases.bottom_up_total_sec =
            non_bookkeeping + phases.direct_bookkeeping_measured_sec;
        if non_bookkeeping + phases.direct_bookkeeping_measured_sec <= ref_sec {
            phases.reference_evolve_wall_sec = ref_sec;
            phases.bookkeeping_and_copies_sec = ref_sec - non_bookkeeping;
        } else {
            let scale = ref_sec / phases.bottom_up_total_sec.max(1.0e-12);
            phases.eos_sec *= scale;
            phases.kap_sec *= scale;
            phases.neu_sec *= scale;
            phases.net_sec *= scale;
            phases.net_and_jacobian_sec *= scale;
            phases.turbulence_sec *= scale;
            phases.equation_assembly_sec *= scale;
            phases.block_solve_sec *= scale;
            phases.remesh_sec *= scale;
            phases.bookkeeping_and_copies_sec =
                phases.direct_bookkeeping_measured_sec * scale;
            phases.reference_evolve_wall_sec = ref_sec;
        }
        Some(phases)
    } else {
        None
    };

    Ok(CaseProfileRecord {
        case_name,
        numerics_mode: mode_str,
        rayon_threads,
        allow_unsupported_controls,
        n_controls_ignored,
        ignored_controls,
        run_mode,
        net_name: final_model.job.net_name.clone(),
        termination_code,
        warmup_wall_sec,
        init_wall_sec,
        evolve_wall_sec,
        total_wall_sec,
        steps_taken,
        total_zone_steps: extras.total_zone_steps,
        total_zone_newton_iters: extras.total_zone_newton_iters,
        initial_zones,
        final_zones: final_model.nz,
        species: final_model.species,
        nvar_coupled,
        total_newton_iterations,
        accepted_step_newton_iterations,
        total_solver_calls,
        total_retries,
        total_redos,
        remesh_applied_steps: extras.remesh_applied_steps,
        micro_cache_hits,
        micro_cache_misses,
        photo_checksum,
        star_model_heap_bytes,
        primary_state_buffer_heap_bytes,
        current_rss_kib: current_rss_kib(),
        peak_rss_kib: peak_rss_kib(),
        final_history,
        phases: phases_opt,
    })
}
