// SPDX-License-Identifier: GPL-3.0-or-later
//! High-level element diffusion driver and post-step diffusion energy
//! accounting ported from MESA `26.4.1` `star/private/element_diffusion.f90`.

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AMU, MixingType, PI4, QE};
use flexmesa_eos::I_PHASE;

use crate::controls_diffusion::MAX_NUM_DIFFUSION_CLASSES;
use crate::diffusion::{
    do_solve_diffusion, set_diffusion_classes, DiffusionScheme,
    SolveDiffusionInput, SolveDiffusionOutput,
};
use crate::job_controls::StarError;
use crate::model::StarModel;

/// Returns the first 0-based species index achieving the maximum mass fraction
/// in `xa_zone`, matching Fortran `maxloc(s% xa(:, k), dim=1)`.
#[inline]
fn maxloc_species(xa_zone: &[f64]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_val = xa_zone.first().copied().unwrap_or(0.0);
    for (j, &val) in xa_zone.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = j;
        }
    }
    best_idx
}

/// Executes one element diffusion step on `model` over timestep `dt_in` (s),
/// matching `star/private/element_diffusion.f90:34-569`.
///
/// Updates `model.xa`, `model.energy_start`, `model.aux.eps_diffusion` (zeroed
/// here and populated after `set_hydro_vars` by [`finish_element_diffusion`]),
/// `model.aux.eps_wd_sedimentation`, `model.aux.edv`, `model.aux.e_field`,
/// `model.aux.g_field_element_diffusion`, `model.aux.diffusion_d_self`,
/// `model.aux.v_rad`, `model.aux.g_rad`, `model.aux.typical_charge`,
/// `model.aux.diffusion_dx`, and the diffusion solver step/iteration counters.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] for incompatible heating controls,
/// unsupported `use_other_diffusion*` hooks, missing `h1`/`he4` isotopes,
/// unsupported radiative levitation (`op_mono`, `#286`), an active
/// `diffusion_dump_call_number` debug stop (`element_diffusion.f90:242,293,361`),
/// or the upstream out-of-bounds boundary conditions tracked in `#272`, `#287`,
/// and `#288`, and [`StarError::ConvergenceFailure`] if the coupled diffusion
/// solver fails to complete `dt_in` (restoring `model.xa` to its pre-diffusion
/// state).
#[allow(clippy::too_many_lines, clippy::similar_names)]
pub fn do_element_diffusion_step(
    model: &mut StarModel,
    dt_in: f64,
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;

    model.aux.step_hist.num_diffusion_solver_steps = 0;
    model.aux.eps_diffusion.resize(nz, 0.0);
    model.aux.eps_diffusion.fill(0.0);
    if model.energy_start.len() != nz {
        model.energy_start.resize(nz, 0.0);
    }
    model.energy_start[..nz].copy_from_slice(&model.energy[..nz]);

    let dctrl = model.controls.diffusion.clone();
    if !model.controls.do_element_diffusion || dt_in < dctrl.diffusion_dt_limit
    {
        model.aux.step_hist.num_diffusion_solver_iters = 0;
        model.aux.edv.resize(nz * species, 0.0);
        model.aux.edv.fill(0.0);
        model.aux.eps_wd_sedimentation.resize(nz, 0.0);
        model.aux.eps_wd_sedimentation.fill(0.0);
        return Ok(());
    }

    if dctrl.use_other_diffusion {
        return Err(StarError::InvalidInput(
            "use_other_diffusion = .true. is not supported in flexmesa-star"
                .into(),
        ));
    }
    if dctrl.use_other_diffusion_factor {
        return Err(StarError::InvalidInput(
            "use_other_diffusion_factor = .true. is not supported in flexmesa-star"
                .into(),
        ));
    }
    if dctrl.use_other_diffusion_coefficients {
        return Err(StarError::InvalidInput(
            "use_other_diffusion_coefficients = .true. is not supported in flexmesa-star"
                .into(),
        ));
    }
    if dctrl.do_diffusion_heating && dctrl.do_wd_sedimentation_heating {
        return Err(StarError::InvalidInput(
            "do_diffusion_heating is incompatible with do_WD_sedimentation_heating; \
             at least one of these options must be set to .false."
                .into(),
        ));
    }

    if nz == 0 || species == 0 {
        return Err(StarError::InvalidInput(
            "element_diffusion reached upstream out-of-bounds empty model \
             (nz == 0 or species == 0; tracked under #287)"
                .into(),
        ));
    }

    let nc = if dctrl.diffusion_use_full_net {
        if species > MAX_NUM_DIFFUSION_CLASSES {
            return Err(StarError::InvalidInput(format!(
                "net is too large for diffusion_use_full_net ({species} > \
                 max_num_diffusion_classes = {MAX_NUM_DIFFUSION_CLASSES})"
            )));
        }
        species
    } else {
        // Upstream `element_diffusion.f90:123` copies `diffusion_num_classes`
        // without range checking, indexing past `diffusion_class_representative(100)`
        // or writing `X(0, k)` when `nc == 0` (#287).
        if dctrl.diffusion_num_classes == 0
            || dctrl.diffusion_num_classes > MAX_NUM_DIFFUSION_CLASSES
        {
            return Err(StarError::InvalidInput(format!(
                "invalid diffusion_num_classes = {} (must be in \
                 1..={MAX_NUM_DIFFUSION_CLASSES}; upstream OOB tracked under #287)",
                dctrl.diffusion_num_classes
            )));
        }
        dctrl.diffusion_num_classes
    };

    // set_extras (element_diffusion.f90:537-566)
    let mut free_e = vec![0.0_f64; nz];
    let mut gamma = vec![0.0_f64; nz];
    let mut dlnp_dm = vec![0.0_f64; nz];
    let mut dlnt_dm = vec![0.0_f64; nz];
    for k in 0..nz {
        free_e[k] = flexmesa_math::exp(model.lnfree_e[k]);
        gamma[k] = model.gam[k];
        if k > 0 {
            let r_k = model.r[k];
            let grav = -model.cgrav[k] * model.m[k] / (r_k * r_k);
            let area = PI4 * (r_k * r_k);
            let p_face = 0.5 * (model.peos[k] + model.peos[k - 1]);
            dlnp_dm[k] = grav / (area * p_face);
            dlnt_dm[k] = model.gradt[k] * dlnp_dm[k];
        }
    }

    // Active region bounds nzlo..=nzhi (element_diffusion.f90:144-219)
    let min_d_mix = dctrl.d_mix_ignore_diffusion;
    let mut nzlo = 0_usize;
    if dctrl.diffusion_min_t_at_surface > 0.0 {
        let mut k = nzlo;
        while model.t[k] < dctrl.diffusion_min_t_at_surface {
            k += 1;
            if k >= nz {
                break;
            }
            nzlo = k;
        }
    }

    if dctrl.diffusion_min_dq_ratio_at_surface > 0.0 {
        let mut dqsum: f64 = model.dq[..=nzlo].iter().sum();
        let mut k = nzlo;
        // Upstream `element_diffusion.f90:159` reads `s% dq(k+1)` on entry,
        // which is out of bounds when `nzlo == nz` (0-based `nz - 1`, #287).
        if k + 1 >= nz {
            return Err(StarError::InvalidInput(
                "element_diffusion reached upstream out-of-bounds dq(nz+1) \
                 read in diffusion_min_dq_ratio_at_surface loop \
                 (nzlo reached innermost zone; tracked under #287)"
                    .into(),
            ));
        }
        while dqsum < dctrl.diffusion_min_dq_ratio_at_surface * model.dq[k + 1]
        {
            k += 1;
            if k + 1 >= nz {
                break;
            }
            dqsum += model.dq[k];
            nzlo = k;
        }
    }

    if dctrl.diffusion_min_dq_at_surface > 0.0 {
        let mut dqsum: f64 = model.dq[..=nzlo].iter().sum();
        let mut k = nzlo;
        while dqsum < dctrl.diffusion_min_dq_at_surface {
            k += 1;
            if k >= nz {
                break;
            }
            if maxloc_species(&model.xa[k * species..(k + 1) * species])
                != maxloc_species(&model.xa[(k - 1) * species..k * species])
            {
                break;
            }
            dqsum += model.dq[k];
            nzlo = k;
        }
    }

    if model.d_mix[nzlo] > min_d_mix {
        if nzlo == 0 {
            // Upstream MESA 26.4.1 `element_diffusion.f90:185-194` indexes
            // `s% xa(:, k - 1)` starting at `k = nzlo = 1` (out-of-bounds
            // `xa(:, 0)`). Fail loudly instead of reading OOB memory (#272).
            return Err(StarError::InvalidInput(
                "element_diffusion reached upstream out-of-bounds xa(:, 0) \
                 condition (nzlo == 0 with D_mix[0] > D_mix_ignore_diffusion; \
                 tracked under #272)"
                    .into(),
            ));
        }
        let kk = nzlo;
        for k in nzlo..(nz - 1) {
            if maxloc_species(&model.xa[k * species..(k + 1) * species])
                != maxloc_species(&model.xa[(k - 1) * species..k * species])
                || model.d_mix[k + 1] <= min_d_mix
            {
                nzlo = k;
                break;
            }
        }
        nzlo = kk + (nzlo - kk) * 4 / 5;
    }

    let mut nzhi = nz - 1;
    if model.d_mix[nzhi] > min_d_mix {
        for k in (1..nz).rev() {
            if model.d_mix[k - 1] <= min_d_mix {
                nzhi = k;
                break;
            }
        }
        nzhi = (3 * nzhi + (nz - 1)) / 4;
    }

    if model.controls.do_phase_separation
        && dctrl.phase_separation_no_diffusion
    {
        for k in 0..=nzhi {
            if model.mixing_type[k] == MixingType::PhaseSeparationMixing {
                nzhi = k;
                break;
            }
        }
    }

    // Resolve diffusion class representatives (element_diffusion.f90:221-235).
    let catalog = NuclideCatalog::builtin_ref();
    let mut class_chem_id = vec![0_usize; nc];
    let mut class_a = vec![0.0_f64; nc];
    let mut class_z = vec![0_usize; nc];

    if dctrl.diffusion_use_full_net {
        for j in 0..nc {
            let cid = model.chem_id[j];
            let iso = catalog.get_by_id(cid).ok_or_else(|| {
                StarError::InvalidInput(format!(
                    "unknown network chem_id {cid} in element diffusion"
                ))
            })?;
            class_chem_id[j] = cid;
            class_a[j] = (iso.z + iso.n) as f64;
            class_z[j] = iso.z;
        }
    } else {
        for j in 0..nc {
            let name = dctrl
                .diffusion_class_representative
                .get(j)
                .map_or("", String::as_str)
                .trim();
            let Some(iso) = catalog.get_by_name(name) else {
                // Upstream MESA 26.4.1 `element_diffusion.f90:226-234` prints
                // a diagnostic and returns early with `ierr = 0` without
                // mutating `xa` when a class representative name is invalid
                // (#273). Reproduce verbatim.
                return Ok(());
            };
            class_chem_id[j] = iso.chem_id;
            class_a[j] = (iso.z + iso.n) as f64;
            class_z[j] = iso.z;
        }
    }

    let mut species_a = vec![0.0_f64; species];
    let mut species_z = vec![0_usize; species];
    for (j, &cid) in model.chem_id.iter().enumerate() {
        let iso = catalog.get_by_id(cid).ok_or_else(|| {
            StarError::InvalidInput(format!(
                "unknown network chem_id {cid} in element diffusion"
            ))
        })?;
        species_a[j] = (iso.z + iso.n) as f64;
        species_z[j] = iso.z;
    }
    let class = set_diffusion_classes(
        nc,
        &species_a,
        &dctrl.diffusion_class_a_max,
        dctrl.diffusion_use_full_net,
    );

    model.aux.diffusion_call_number += 1;
    // Upstream `element_diffusion.f90:242, 293, 361` writes `diffusion.data`
    // and terminates via `mesa_error('debug: dump_diffusion_info')` when
    // `diffusion_call_number == diffusion_dump_call_number`.
    if (model.aux.diffusion_call_number as isize)
        == dctrl.diffusion_dump_call_number
    {
        return Err(StarError::InvalidInput(format!(
            "debug: dump_diffusion_info triggered at diffusion_call_number = {} \
             (diffusion_dump_call_number = {}; \
             `star/private/element_diffusion.f90:242,293,361`)",
            model.aux.diffusion_call_number, dctrl.diffusion_dump_call_number
        )));
    }

    let calculate_ionization =
        dctrl.diffusion_calculates_ionization || dctrl.diffusion_use_full_net;
    let mut typical_charge = vec![0.0_f64; nz * nc];
    if !calculate_ionization {
        for k in 0..nz {
            for j in 0..nc {
                typical_charge[k * nc + j] = dctrl
                    .diffusion_class_typical_charge
                    .get(j)
                    .copied()
                    .unwrap_or(1.0);
            }
        }
    }

    let xa_save = model.xa.clone();

    // Face log-gradients (element_diffusion.f90:280-291).
    let mut dln_p_dm_face = vec![0.0_f64; nz];
    let mut dln_t_dm_face = vec![0.0_f64; nz];
    let mut dln_rho_dm_face = vec![0.0_f64; nz];
    for k in 1..nz {
        dln_p_dm_face[k] = dlnp_dm[k];
        dln_t_dm_face[k] = dlnt_dm[k];
        let alfa = model.dm[k - 1] / (model.dm[k] + model.dm[k - 1]);
        let chi_t_face =
            alfa * model.chi_t[k] + (1.0 - alfa) * model.chi_t[k - 1];
        let chi_rho_face =
            alfa * model.chi_rho[k] + (1.0 - alfa) * model.chi_rho[k - 1];
        let grad_rho_face = (1.0 - chi_t_face * model.gradt[k]) / chi_rho_face;
        dln_rho_dm_face[k] = grad_rho_face * dlnp_dm[k];
    }

    let phase: Vec<f64> =
        model.eos_res.iter().map(|res| res[I_PHASE]).collect();
    let has_h1 = catalog
        .get_by_name("h1")
        .and_then(|iso| model.net_iso.get(iso.chem_id).copied().flatten())
        .is_some();
    let has_he4 = catalog
        .get_by_name("he4")
        .and_then(|iso| model.net_iso.get(iso.chem_id).copied().flatten())
        .is_some();

    let max_steps = model
        .controls
        .ts_stop
        .diffusion_steps_hard_limit
        .max(0) as usize;
    let max_iters_total = model
        .controls
        .ts_stop
        .diffusion_iters_hard_limit
        .max(0) as usize;

    let diffusion_factor_vec: Vec<f64> = (0..nc)
        .map(|j| dctrl.diffusion_class_factor.get(j).copied().unwrap_or(1.0))
        .collect();
    let mut x_init = vec![0.0_f64; nz * nc];
    let mut x_final = vec![0.0_f64; nz * nc];
    let mut d_self = vec![0.0_f64; nz * nc];
    let mut v_advection = vec![0.0_f64; nz * nc];
    let mut v_total = vec![0.0_f64; nz * nc];
    let mut vln_p = vec![0.0_f64; nz * nc];
    let mut vln_t = vec![0.0_f64; nz * nc];
    let mut v_rad = vec![0.0_f64; nz * nc];
    let mut g_rad = vec![0.0_f64; nz * nc];
    let mut e_field = vec![0.0_f64; nz];
    let mut g_field = vec![0.0_f64; nz];

    let solve_inp = SolveDiffusionInput {
        nz,
        species,
        nc,
        class: &class,
        class_a: &class_a,
        class_z: &class_z,
        has_h1,
        has_he4,
        abar: &model.abar,
        free_e: &free_e,
        dm_bar_in: &model.dm_bar,
        dm_in: &model.dm,
        t: &model.t,
        ln_t: &model.ln_t,
        rho: &model.rho,
        lnd: &model.lnd,
        r_mid: &model.rmid,
        r_face_in: &model.r,
        dln_p_dm_face: &dln_p_dm_face,
        dln_t_dm_face: &dln_t_dm_face,
        dln_rho_dm_face: &dln_rho_dm_face,
        grav: &model.grav,
        eta: &model.eta,
        gamma: &gamma,
        phase: &phase,
        r_center: model.r_center,
        star_dt: model.dt,
        total_time: dt_in,
        pure_coulomb: dctrl.diffusion_use_iben_macdonald,
        diffusion_use_paquette: dctrl.diffusion_use_paquette,
        diffusion_use_caplan: dctrl.diffusion_use_caplan,
        diffusion_use_cgs_solver: dctrl.diffusion_use_cgs_solver,
        cgs_thermal_diffusion_eta_full_on: dctrl
            .cgs_thermal_diffusion_eta_full_on,
        cgs_thermal_diffusion_eta_full_off: dctrl
            .cgs_thermal_diffusion_eta_full_off,
        dt_div_timescale: dctrl.diffusion_dt_div_timescale,
        min_num_substeps: dctrl.diffusion_min_num_substeps,
        max_steps,
        max_iters_total,
        max_iters_per_substep: dctrl.diffusion_max_iters_per_substep,
        max_retries_per_substep: dctrl.diffusion_max_retries_per_substep,
        calculate_ionization,
        nsmooth_typical_charge: dctrl.diffusion_nsmooth_typical_charge,
        min_t_for_radaccel: dctrl.diffusion_min_t_for_radaccel,
        max_t_for_radaccel: dctrl.diffusion_max_t_for_radaccel,
        v_advection_max: dctrl.diffusion_v_max,
        diffusion_vlimit: dctrl.diffusion_vlimit,
        diffusion_vlimit_dm_full_on: dctrl.diffusion_vlimit_dm_full_on,
        diffusion_vlimit_dm_full_off: dctrl.diffusion_vlimit_dm_full_off,
        diffusion_ad_dm_full_on: dctrl.diffusion_ad_dm_full_on,
        diffusion_ad_dm_full_off: dctrl.diffusion_ad_dm_full_off,
        diffusion_ad_boost_factor: dctrl.diffusion_ad_boost_factor,
        diffusion_sig_factor: dctrl.diffusion_sig_factor,
        diffusion_gt_factor: dctrl.diffusion_gt_factor,
        upwind_limit: dctrl.diffusion_upwind_abs_v_limit,
        gamma_full_on: dctrl.diffusion_gamma_full_on,
        gamma_full_off: dctrl.diffusion_gamma_full_off,
        t_full_on: dctrl.diffusion_t_full_on,
        t_full_off: dctrl.diffusion_t_full_off,
        min_x_hard_limit: dctrl.diffusion_min_x_hard_limit,
        x_total_atol: dctrl.diffusion_x_total_atol,
        x_total_rtol: dctrl.diffusion_x_total_rtol,
        tol_correction_max: dctrl.diffusion_tol_correction_max,
        tol_correction_norm: dctrl.diffusion_tol_correction_norm,
        diffusion_factor: &diffusion_factor_vec,
        use_other_diffusion_factor: false,
        extra_diffusion_factor: None,
        use_isolve: dctrl.diffusion_use_isolve,
        rtol_for_isolve: dctrl.diffusion_rtol_for_isolve,
        atol_for_isolve: dctrl.diffusion_atol_for_isolve,
        maxsteps_for_isolve: dctrl.diffusion_maxsteps_for_isolve,
        isolve_solver_name: &dctrl.diffusion_isolve_solver,
        nzlo_in: nzlo,
        nzhi_in: nzhi,
    };

    let solve_res = {
        let mut solve_out = SolveDiffusionOutput {
            typical_charge: &mut typical_charge,
            xa: &mut model.xa,
            x_init: &mut x_init,
            x_final: &mut x_final,
            d_self_face: &mut d_self,
            v_advection_face: &mut v_advection,
            v_total_face: &mut v_total,
            vln_p_face: &mut vln_p,
            vln_t_face: &mut vln_t,
            v_rad_face: &mut v_rad,
            g_rad_face: &mut g_rad,
            e_field_face: &mut e_field,
            g_field_face: &mut g_field,
        };
        do_solve_diffusion(&solve_inp, &mut solve_out)
    };

    let stats = match solve_res {
        Ok(st) => st,
        Err(err) => {
            model.xa = xa_save;
            return Err(err);
        }
    };

    model.aux.step_hist.num_diffusion_solver_steps = stats.steps_used;
    model.aux.step_hist.num_diffusion_solver_iters = stats.total_num_iters;

    let nzlo_eff = stats.nzlo;
    let nzhi_eff = stats.nzhi;

    model.aux.diffusion_d_self.resize(nz * species, 0.0);
    model.aux.edv.resize(nz * species, 0.0);
    model.aux.v_rad.resize(nz * species, 0.0);
    model.aux.g_rad.resize(nz * species, 0.0);
    model.aux.typical_charge.resize(nz * species, 0.0);
    model.aux.diffusion_dx.resize(nz * species, 0.0);
    model.aux.e_field.resize(nz, 0.0);
    model.aux.g_field_element_diffusion.resize(nz, 0.0);
    model.aux.eps_wd_sedimentation.resize(nz, 0.0);
    model.aux.eps_wd_sedimentation.fill(0.0);

    // Copy active-face electric and gravitational fields from do_solve_diffusion
    // (upstream passes `s% E_field` and `s% g_field_element_diffusion` directly
    // into `do_solve_diffusion`, which writes `(nzlo_eff + 1)..=nzhi_eff`).
    if nzlo_eff < nzhi_eff {
        model.aux.e_field[(nzlo_eff + 1)..=nzhi_eff]
            .copy_from_slice(&e_field[(nzlo_eff + 1)..=nzhi_eff]);
        model.aux.g_field_element_diffusion[(nzlo_eff + 1)..=nzhi_eff]
            .copy_from_slice(&g_field[(nzlo_eff + 1)..=nzhi_eff]);
    }

    // Post-solve diagnostic unpack and surface/center extrapolation
    // (`element_diffusion.f90:363-434`). When `nzhi_eff <= nzlo_eff`, the
    // active loops run zero iterations and the boundary loops still execute.
    for k in (nzlo_eff + 1)..=nzhi_eff {
        for (j, &cls) in class.iter().enumerate().take(species) {
            model.aux.diffusion_d_self[k * species + j] = d_self[k * nc + cls];
            model.aux.edv[k * species + j] = v_total[k * nc + cls];
            model.aux.v_rad[k * species + j] = v_rad[k * nc + cls];
            model.aux.g_rad[k * species + j] = g_rad[k * nc + cls];
            model.aux.typical_charge[k * species + j] =
                typical_charge[k * nc + cls];
            model.aux.diffusion_dx[k * species + j] =
                model.xa[k * species + j] - xa_save[k * species + j];
        }
    }

    if dctrl.do_wd_sedimentation_heating {
        let eps_fac = model.controls.mix_wind.eps_wd_sedimentation_factor;
        let min_xa = dctrl.min_xa_for_wd_sedimentation_heating;
        let v_max = dctrl.diffusion_v_max;
        for k in (nzlo_eff + 1)..=nzhi_eff {
            let mut eps_k = 0.0;
            let e_k = model.aux.e_field[k];
            let g_k = model.aux.g_field_element_diffusion[k];
            for i in 0..species {
                let x_ik = model.xa[k * species + i];
                if x_ik > min_xa {
                    let amass = species_a[i];
                    let zcharge = species_z[i] as f64;
                    let edv_ik = model.aux.edv[k * species + i];
                    let sgn = 1.0_f64.copysign(-edv_ik);
                    eps_k += eps_fac
                        * (amass - zcharge * QE * e_k / (AMU * g_k))
                        * sgn
                        * v_max.min(edv_ik.abs())
                        * x_ik
                        * g_k
                        / amass;
                }
            }
            model.aux.eps_wd_sedimentation[k] = eps_k;
        }
    }

    let k_ref_top = nzlo_eff + 1;
    if k_ref_top >= nz {
        // Upstream `element_diffusion.f90:410-421` indexes `nzlo + 1` even when
        // `nzlo == nz` (0-based `nz - 1`), reading past array bounds (#287).
        return Err(StarError::InvalidInput(
            "element_diffusion reached upstream out-of-bounds read at \
             nzlo + 1 == nz + 1 in post-solve diagnostic fill \
             (tracked under #287)"
                .into(),
        ));
    }
    for k in 0..=nzlo_eff {
        for j in 0..species {
            model.aux.diffusion_d_self[k * species + j] =
                model.aux.diffusion_d_self[k_ref_top * species + j];
            model.aux.edv[k * species + j] = 0.0;
            model.aux.v_rad[k * species + j] =
                model.aux.v_rad[k_ref_top * species + j];
            model.aux.g_rad[k * species + j] =
                model.aux.g_rad[k_ref_top * species + j];
            model.aux.typical_charge[k * species + j] =
                model.aux.typical_charge[k_ref_top * species + j];
            model.aux.diffusion_dx[k * species + j] =
                model.xa[k * species + j] - xa_save[k * species + j];
        }
        model.aux.e_field[k] = 0.0;
        model.aux.g_field_element_diffusion[k] = 0.0;
    }

    for k in (nzhi_eff + 1)..nz {
        for j in 0..species {
            model.aux.diffusion_d_self[k * species + j] =
                model.aux.diffusion_d_self[nzhi_eff * species + j];
            model.aux.edv[k * species + j] = 0.0;
            model.aux.v_rad[k * species + j] =
                model.aux.v_rad[nzhi_eff * species + j];
            model.aux.g_rad[k * species + j] =
                model.aux.g_rad[nzhi_eff * species + j];
            model.aux.typical_charge[k * species + j] =
                model.aux.typical_charge[nzhi_eff * species + j];
            model.aux.diffusion_dx[k * species + j] =
                model.xa[k * species + j] - xa_save[k * species + j];
        }
        model.aux.e_field[k] = 0.0;
        model.aux.g_field_element_diffusion[k] = 0.0;
    }

    Ok(())
}

/// Computes `eps_diffusion(k) = (energy_start(k) - energy(k)) / dt` after
/// post-diffusion `set_hydro_vars` updates EOS internal energy
/// (`element_diffusion.f90:571-580`).
pub fn finish_element_diffusion(model: &mut StarModel, dt: f64) {
    let nz = model.nz;
    model.aux.eps_diffusion.resize(nz, 0.0);
    for k in 0..nz {
        model.aux.eps_diffusion[k] =
            (model.energy_start[k] - model.energy[k]) / dt;
    }
}

/// Compatibility entry point executing real MESA `26.4.1` element diffusion
/// with the requested collision-integral `scheme` and returning
/// `(edv, eps_wd_sedimentation)`.
///
/// # Errors
///
/// Propagates any [`StarError`] returned by [`do_element_diffusion_step`].
pub fn do_element_diffusion(
    model: &mut StarModel,
    dt: f64,
    scheme: DiffusionScheme,
) -> Result<(Vec<f64>, Vec<f64>), StarError> {
    let saved_do = model.controls.do_element_diffusion;
    let saved_diff = model.controls.diffusion.clone();

    model.controls.do_element_diffusion = true;
    model.controls.diffusion.diffusion_dt_limit = 0.0;
    model.controls.diffusion.diffusion_use_full_net = true;
    model.controls.diffusion.diffusion_use_cgs_solver = true;
    model.controls.diffusion.diffusion_use_iben_macdonald =
        scheme == DiffusionScheme::IbenMacDonald;
    model.controls.diffusion.diffusion_use_paquette =
        scheme == DiffusionScheme::Paquette;
    model.controls.diffusion.diffusion_use_caplan =
        scheme == DiffusionScheme::CaplanOcp;

    let res = do_element_diffusion_step(model, dt);

    model.controls.do_element_diffusion = saved_do;
    model.controls.diffusion = saved_diff;

    res?;
    Ok((
        model.aux.edv.clone(),
        model.aux.eps_wd_sedimentation.clone(),
    ))
}
