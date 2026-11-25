// SPDX-License-Identifier: GPL-3.0-or-later
//! The surface removal that relaxes to a trimmed model, at MESA `26.4.1`:
//! `do_relax_to_star_cut` (`star/private/remove_shells.f90:1156-1404`), reached
//! from `do_remove_surface` when `remove_surface_by_relax_to_star_cut` is set
//! (`remove_shells.f90:1024-1035`) and from `star_relax_to_star_cut`
//! (`star/public/star_lib.f90:2540-2550`).
//!
//! The routine saves the composition and entropy profiles of the cells from
//! `k_remove` inward, loads a ZAMS model of the mass interior to that cell on
//! `basic.net`, restores the net, and relaxes the new model onto the saved
//! profiles with `do_relax_composition` and `do_relax_entropy`.

use std::path::Path;

use flexmesa_const::{AVO, KERG, M_SUN};

use crate::controls::StarError;
use crate::inlist_runner::StarExtras;
use crate::micro::change_model_network_with_adjust;
use crate::model::StarModel;
use crate::relax_comp::CompositionProfile;
use crate::relax_entropy::{RelaxEntropyTarget, do_relax_entropy};
use crate::relax_mass_comp::do_relax_composition;
use crate::star_job_init::prepare_new_model_for_relax;
use crate::zams::{ZamsCatalog, load_zams_model_from_catalog};

/// Net that the ZAMS model is loaded on (`remove_shells.f90:1303`).
const ZAMS_NET: &str = "basic.net";

/// `do_remove_surface_at_cell_k` (`star/private/remove_shells.f90:729-740`),
/// the routine behind `star_remove_surface_at_cell_k`, followed by
/// `do_remove_surface` (`remove_shells.f90:996-1035`): removes the cells
/// outside `k_remove` (1-based; the cell `k_remove` becomes the outermost).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when `remove_surface_by_relax_to_star_cut`
/// is not set, because only the relaxing removal is ported (`#153` tracks the
/// mesh-pruning removal), and the errors of [`do_relax_to_star_cut`].
pub fn remove_surface_at_cell_k(
    model: &mut StarModel,
    k_remove: usize,
    search_dir: Option<&Path>,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    // `if (k <= 1) return` (`remove_shells.f90:738`), and
    // `surface_k == 1` in `do_remove_surface` (`remove_shells.f90:1012-1015`).
    if k_remove <= 1 {
        return Ok(());
    }
    if !model.job.remove_surface_by_relax_to_star_cut {
        return Err(StarError::InvalidInput(String::from(
            "do_remove_surface without remove_surface_by_relax_to_star_cut \
             (do_prune_mesh_surface, star/private/remove_shells.f90:1037-1149) \
             is not ported (#153)",
        )));
    }
    // `remove_shells.f90:1024-1030`: upstream sets `ierr = -1` and stops.
    if model.r_center != 0.0 {
        return Err(StarError::InvalidInput(String::from(
            "remove surface currently requires model with inner boundary at \
             true center of star (star/private/remove_shells.f90:1025-1029)",
        )));
    }
    let (do_jrot, do_entropy, turn_off) = (
        model.job.remove_surface_do_jrot,
        model.job.remove_surface_do_entropy,
        model.job.remove_surface_turn_off_energy_sources_and_sinks,
    );
    do_relax_to_star_cut(
        model, k_remove, do_jrot, do_entropy, turn_off, search_dir, extras,
    )
}

/// `do_relax_to_star_cut` (`star/private/remove_shells.f90:1156-1404`):
/// relaxes `model` to a model whose outermost cell is the current cell
/// `k_remove` (1-based), keeping the composition and, with `do_entropy`, the
/// entropy of the cells that remain.
///
/// `turn_off_energy_sources_and_sinks` zeroes `non_nuc_neu_factor` and
/// `eps_nuc_factor` during the relaxes. The history trace count
/// (`num_trace_history_values`) is not modeled, and the ZAMS model is built
/// fresh rather than reloaded into the old arrays, so state that upstream
/// leaves in place (`star_age`, `dt_next`, the solver counters, the energy
/// error and heating sums) is carried over by hand.
///
/// As upstream, the velocity flags are restored afterwards, but
/// `rotation_flag` and `RTI_flag` are not (`remove_shells.f90:1271-1280,
/// 1328-1330, 1375-1382`; `RTI_flag`: #533), and `v` and `u` restart at zero.
///
/// # Errors
/// Returns [`StarError::InvalidInput`] when `k_remove` is outside `1..=nz`
/// (upstream would index outside the arrays), when `do_jrot` is set for a
/// rotating model (`do_relax_angular_momentum` is not ported, `#436`), when
/// `initial_mass` of the cut lies outside the ZAMS catalog
/// (`get_zams_model` would run `do_relax_mass`, `#154`), and the errors of
/// the net change and the two relaxes.
pub fn do_relax_to_star_cut(
    model: &mut StarModel,
    k_remove: usize,
    do_jrot: bool,
    do_entropy: bool,
    turn_off_energy_sources_and_sinks: bool,
    search_dir: Option<&Path>,
    extras: &mut dyn StarExtras,
) -> Result<(), StarError> {
    let nz = model.nz;
    if k_remove < 1 || k_remove > nz {
        return Err(StarError::InvalidInput(format!(
            "do_relax_to_star_cut: k_remove = {k_remove} is outside 1..={nz} \
             (star/private/remove_shells.f90:1206-1222)"
        )));
    }
    if do_jrot && model.controls.rotation_flag {
        return Err(StarError::InvalidInput(String::from(
            "do_relax_to_star_cut with do_jrot on a rotating model needs \
             do_relax_angular_momentum \
             (star/private/remove_shells.f90:1328-1341), which is not ported \
             (#436)",
        )));
    }
    let kr = k_remove - 1;
    let species = model.species;
    let num_pts = nz - kr;

    // `remove_shells.f90:1191-1205`.
    let eps_nuc_factor = model.controls.eps_nuc_factor;
    let non_nuc_neu_factor = model.controls.non_nuc_neu_factor;
    let net_name = model.job.net_name.clone();
    let time = model.time_sec;
    let hist_time = model.aux.step_hist.time_sec;
    let model_number = model.model_number;
    let cumulative_energy_error = model.aux.cumulative_energy_error;
    let cumulative_extra_heating = model.aux.step_hist.cumulative_extra_heating;

    // Cell-centered exterior mass fraction of the remaining mass
    // (`remove_shells.f90:1218-1222`).
    let q_cut = model.q[kr];
    let mut xq = Vec::with_capacity(num_pts);
    xq.push(model.dq[kr] / 2.0 / q_cut);
    for j in 1..num_pts {
        xq.push(
            xq[j - 1] + (model.dq[kr + j] + model.dq[kr + j - 1]) / q_cut / 2.0,
        );
    }
    // `xa(:,:) = s% xa(:,k_remove:s% nz)` and `entropy(:) = s% entropy(...)
    // *avo*kerg` (`remove_shells.f90:1247-1255`).
    let xa = model.xa[kr * species..nz * species].to_vec();
    let entropy: Vec<f64> = model.entropy[kr..nz]
        .iter()
        .map(|&s| s * AVO * KERG)
        .collect();
    let profile = CompositionProfile {
        num_pts,
        species,
        xq: xq.clone(),
        xa,
    };
    // Built before the model is touched so that a bad profile leaves it as
    // it was (`interp_pm` errors, `relax.f90:803-804`).
    let entropy_target = if do_entropy {
        Some(RelaxEntropyTarget::new(&xq, &entropy)?)
    } else {
        None
    };

    // The ZAMS model of the mass inside the cut, with the velocity,
    // rotation and RTI flags off (`remove_shells.f90:1257-1281`). `v_flag`
    // and `u_flag` come back below.
    let v_flag = model.controls.v_flag;
    let u_flag = model.controls.u_flag;
    let have_mlt_vc = model.aux.have_mlt_vc;
    let photo_interval = model.controls.output.photo_interval;
    let initial = (
        model.controls.initial_z,
        model.controls.initial_y,
        model.controls.initial_mass,
    );
    let mut controls = model.controls.clone();
    controls.v_flag = false;
    controls.u_flag = false;
    // Upstream saves `RTI_flag` but never restores it (#533).
    controls.rti_flag = false;
    controls.rotation_flag = false;
    controls.output.photo_interval = 10_000_000;
    controls.initial_z = 0.02;
    controls.initial_y = 0.28;
    controls.initial_mass = model.m[kr] / M_SUN;
    let mut job = model.job.clone();
    job.net_name = String::from(ZAMS_NET);
    let mut new =
        load_zams_model_from_catalog(ZamsCatalog::builtin(), controls, job)?;
    if new.controls.initial_mass != model.m[kr] / M_SUN {
        return Err(StarError::InvalidInput(format!(
            "do_relax_to_star_cut: the cut mass {} Msun is outside the ZAMS \
             catalog, where get_zams_model relaxes the mass with \
             do_relax_mass (star/private/init_model.f90:119-126, #154)",
            model.m[kr] / M_SUN
        )));
    }

    // State that upstream keeps in `s` across `load_zams_model`.
    new.eos_config = model.eos_config;
    new.kap_config = model.kap_config.clone();
    new.star_age = model.star_age;
    new.star_age_old = model.star_age_old;
    new.total_num_solver_iterations = model.total_num_solver_iterations;
    new.total_num_solver_calls = model.total_num_solver_calls;
    // `dt_next` stays unless it is not positive, and `model_builder` then
    // limits it by `max_years_for_timestep` (`init.f90:1018-1022, 1068-1069`).
    if model.dt_next > 0.0 {
        new.dt_next = model.dt_next;
    }
    new.dt = 0.0;
    if new.controls.max_years_for_timestep > 0.0 {
        new.dt_next = new
            .dt_next
            .min(flexmesa_const::SECYER * new.controls.max_years_for_timestep);
    }
    // `s% model_number = 0` and `s% time = 0` for the relaxes
    // (`remove_shells.f90:1203-1205`).
    new.model_number = 0;
    new.time_sec = 0.0;
    new.aux.step_hist.time_sec = 0.0;
    *model = new;

    // `change_net(id, .true., net_name)` (`remove_shells.f90:1309-1311`).
    change_model_network_with_adjust(model, &net_name, search_dir, true)?;
    // `load_zams_model` ends in `finish_load_model`.
    prepare_new_model_for_relax(model, extras)?;
    model.time_sec = 0.0;
    model.aux.step_hist.time_sec = 0.0;
    model.aux.have_mlt_vc = have_mlt_vc;

    if turn_off_energy_sources_and_sinks {
        model.controls.non_nuc_neu_factor = 0.0;
        model.controls.eps_nuc_factor = 0.0;
    }

    do_relax_composition(
        model,
        model.job.num_steps_to_relax_composition,
        &profile,
        extras,
    )?;
    if let Some(target) = entropy_target {
        let max_steps = model.job.max_steps_to_relax_entropy;
        do_relax_entropy(model, max_steps, target, extras)?;
    }

    // `s% generations = 1`, then the velocity flags come back with zero
    // velocities (`set_v_flag`, `set_u_flag`; `remove_shells.f90:1373-1382`).
    model.generations = 1;
    let nz_new = model.nz;
    if v_flag {
        model.controls.v_flag = true;
        model.v = vec![0.0; nz_new];
        model.v_start = vec![0.0; nz_new];
        model.v_center = 0.0;
    }
    if u_flag {
        model.controls.u_flag = true;
        model.u = vec![0.0; nz_new];
        model.u_start = vec![0.0; nz_new];
    }
    model.layout = crate::controls::VarLayout::with_flags(
        model.species,
        model.controls.v_flag,
        model.controls.u_flag,
        model.controls.rsp2_flag,
        model.controls.rti_flag,
    );

    // `remove_shells.f90:1384-1402`.
    model.doing_first_model_of_run = false;
    model.time_sec = time;
    model.aux.step_hist.time_sec = hist_time;
    model.model_number = model_number;
    model.aux.cumulative_energy_error = cumulative_energy_error;
    model.aux.step_hist.cumulative_extra_heating = cumulative_extra_heating;
    model.controls.non_nuc_neu_factor = non_nuc_neu_factor;
    model.controls.eps_nuc_factor = eps_nuc_factor;
    (
        model.controls.initial_z,
        model.controls.initial_y,
        model.controls.initial_mass,
    ) = initial;
    model.controls.output.photo_interval = photo_interval;
    model.aux.need_to_setvars = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::{StarControls, StarJobControls};
    use crate::inlist_runner::DefaultStarExtras;

    /// A 1 Msun ZAMS model whose composition varies smoothly with `q`
    /// (`X_{h1}` raised by `0.1 (1 - q)`, `X_{he4}` lowered to match), set up
    /// as `prepare_new_model_for_relax` leaves a model for a relax.
    fn graded_zams_model() -> StarModel {
        let mut model = crate::zams::load_zams_model(
            StarControls {
                zbase: 0.02,
                ..StarControls::default()
            },
            StarJobControls::default(),
        )
        .expect("1 Msun ZAMS should load");
        let sp = model.species;
        let h1 = model.iso_names.iter().position(|n| n == "h1").unwrap();
        let he4 = model.iso_names.iter().position(|n| n == "he4").unwrap();
        for k in 0..model.nz {
            let d = 0.1 * (1.0 - model.q[k]);
            model.xa[k * sp + h1] += d;
            model.xa[k * sp + he4] -= d;
        }
        model.job.num_steps_to_relax_composition = 2;
        model.job.max_steps_to_relax_entropy = 3;
        model
    }

    fn h1_index(model: &StarModel) -> usize {
        model.iso_names.iter().position(|n| n == "h1").unwrap()
    }

    #[test]
    fn cut_keeps_mass_composition_and_restores_run_state() {
        let mut model = graded_zams_model();
        prepare_new_model_for_relax(&mut model, &mut DefaultStarExtras)
            .expect("prepare");
        model.job.remove_surface_by_relax_to_star_cut = true;
        let kr = 300;
        let m_cut = model.m[kr - 1];
        let q_cut = model.q[kr - 1];
        let h1 = h1_index(&model);
        let sp = model.species;
        let x_h1_cut = model.xa[(kr - 1) * sp + h1];
        let x_h1_center = model.xa[(model.nz - 1) * sp + h1];
        let nz_old = model.nz;
        model.model_number = 7;
        model.star_age = 3.5e6;
        model.time_sec = 11.0;
        model.dt_next = 1.0e3;
        let before = (
            model.model_number,
            model.star_age,
            model.time_sec,
            model.controls.initial_z,
            model.controls.initial_y,
            model.controls.initial_mass,
            model.controls.eps_nuc_factor,
            model.controls.non_nuc_neu_factor,
            model.controls.output.photo_interval,
            model.job.net_name.clone(),
            model.dt_next,
        );

        remove_surface_at_cell_k(&mut model, kr, None, &mut DefaultStarExtras)
            .expect("relax to star cut");

        assert!(model.nz < nz_old);
        // The new model holds the mass interior to the cut.
        assert!((model.mstar - m_cut).abs() <= 1.0e-12 * m_cut);
        assert!(model.m_center == 0.0 && model.r_center == 0.0);
        // `do_relax_composition` sets the new cells to the saved profile.
        let sp_new = model.species;
        assert_eq!(sp_new, sp);
        let x_surf = model.xa[h1];
        let x_cent = model.xa[(model.nz - 1) * sp_new + h1];
        assert!((x_surf - x_h1_cut).abs() <= 2.0e-3, "{x_surf} {x_h1_cut}");
        assert!(
            (x_cent - x_h1_center).abs() <= 2.0e-3,
            "{x_cent} {x_h1_center}"
        );
        assert!(q_cut > 0.99, "the cut keeps nearly all the mass");
        // State restored by `remove_shells.f90:1384-1402` and carried over.
        let after = (
            model.model_number,
            model.star_age,
            model.time_sec,
            model.controls.initial_z,
            model.controls.initial_y,
            model.controls.initial_mass,
            model.controls.eps_nuc_factor,
            model.controls.non_nuc_neu_factor,
            model.controls.output.photo_interval,
            model.job.net_name.clone(),
            model.dt_next,
        );
        assert_eq!(before, after);
        assert!(!model.doing_first_model_of_run);
        assert_eq!(model.generations, 1);
        assert!(model.aux.need_to_setvars);
        assert!(model.aux.relax_entropy.is_none());
    }

    #[test]
    fn cut_without_entropy_relax_skips_the_entropy_hook() {
        let mut model = graded_zams_model();
        prepare_new_model_for_relax(&mut model, &mut DefaultStarExtras)
            .expect("prepare");
        do_relax_to_star_cut(
            &mut model,
            300,
            false,
            false,
            false,
            None,
            &mut DefaultStarExtras,
        )
        .expect("composition-only cut");
        assert!(model.aux.relax_entropy.is_none());
        assert!(!model.controls.use_other_energy);
    }

    #[test]
    fn cut_rejects_bad_requests_and_leaves_the_model_unchanged() {
        let mut model = graded_zams_model();
        let nz = model.nz;
        let run = |model: &mut StarModel, k, jrot| {
            do_relax_to_star_cut(
                model,
                k,
                jrot,
                true,
                true,
                None,
                &mut DefaultStarExtras,
            )
        };
        assert!(run(&mut model, 0, false).is_err());
        assert!(run(&mut model, nz + 1, false).is_err());
        // The innermost cell holds a mass below the ZAMS catalog.
        assert!(run(&mut model, nz, false).is_err());
        model.controls.rotation_flag = true;
        let e = run(&mut model, 300, true).unwrap_err().to_string();
        assert!(e.contains("do_relax_angular_momentum"), "{e}");
        assert_eq!(model.nz, nz);
    }

    #[test]
    fn remove_surface_dispatch_follows_upstream_guards() {
        let mut model = graded_zams_model();
        let nz = model.nz;
        let mut ex = DefaultStarExtras;
        // `k <= 1` returns without a job control (`remove_shells.f90:738`).
        remove_surface_at_cell_k(&mut model, 1, None, &mut ex).unwrap();
        remove_surface_at_cell_k(&mut model, 0, None, &mut ex).unwrap();
        // The mesh-pruning removal is not ported.
        assert!(
            remove_surface_at_cell_k(&mut model, 5, None, &mut ex).is_err()
        );
        model.job.remove_surface_by_relax_to_star_cut = true;
        model.r_center = 1.0e8;
        let e = remove_surface_at_cell_k(&mut model, 5, None, &mut ex)
            .unwrap_err()
            .to_string();
        assert!(e.contains("true center"), "{e}");
        assert_eq!(model.nz, nz);
    }
}
