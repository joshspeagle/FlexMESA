// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone (`#161`/`#152`) microphysics and `do_struct_burn_mix`
//! invariant tests (`micro`, `micro_cache`, `micro_net`, `extend_net`, and
//! `struct_burn_mix`), zero external data dependency.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use flexmesa_chem::NUM_CATEGORIES;
use flexmesa_const::{LN10, SECYER};
use flexmesa_star::extend_net::extend_net;
use flexmesa_star::micro::{
    change_model_network, do_eos, do_net, do_neu, get_or_load_net,
    resolve_model_net, set_basic_vars,
};
use flexmesa_star::micro_net::high_temp_rate_warning_zone;
use flexmesa_star::{
    do_struct_burn_mix, load_zams_model, op_split_burn_zones,
    record_start_of_step_energy_integrals, set_hydro_vars, StarControls,
    StarJobControls,
};

struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir()
            .join(format!("flexmesa_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp test dir");
        Self(dir)
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn default_zams_approx21() -> flexmesa_star::model::StarModel {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    change_model_network(&mut model, "approx21.net", None)
        .expect("switch to approx21.net");
    model.aux.need_to_setvars = true;
    set_hydro_vars(&mut model, false, false).expect("init set_hydro_vars");
    model.aux.need_to_setvars = false;
    model.save_start_values().expect("save_start_values");
    record_start_of_step_energy_integrals(&mut model);
    model.save_old_values();
    model
}

#[test]
fn test_do_struct_burn_mix_op_split_burn_preserves_split_outputs_and_neu() {
    let mut model = default_zams_approx21();
    let nz = model.nz;
    let sp = model.species;
    let kc = nz - 1;
    let i_h1 = model
        .iso_names
        .iter()
        .position(|s| s == "h1")
        .expect("h1 in approx21");

    model.controls.do_mix = false;
    model.controls.op_split_burn = true;
    model.controls.op_split_burn_min_t = 1.0e7;
    model.controls.op_split_burn_min_t_for_variable_t_solver = 1.2e7;
    assert!(flexmesa_math::exp(model.ln_t_start[kc]) >= 1.2e7);

    let xa_pre_step = model.xa.clone();
    let dt = 10.0 * SECYER;
    let rep = do_struct_burn_mix(&mut model, dt, true, true, &[])
        .expect("op_split_burn do_struct_burn_mix");
    assert!(rep.converged, "solver must converge: {rep:?}");
    assert!(!model.aux.need_to_setvars);

    // Split zones must keep burn_avg_epsnuc == eps_nuc, post-burn
    // (xa - xa_pre_step) / dt in dxdt_nuc, and zero coupled partials.
    assert!(model.aux.burn_avg_epsnuc[kc] > 0.0);
    assert_eq!(model.eps_nuc[kc], model.aux.burn_avg_epsnuc[kc]);
    assert_eq!(model.d_epsnuc_dlnd[kc], 0.0);
    assert_eq!(model.d_epsnuc_dlnt[kc], 0.0);
    assert!(model.d_epsnuc_dx[kc * sp..(kc + 1) * sp]
        .iter()
        .all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dlnd[kc * sp..(kc + 1) * sp]
        .iter()
        .all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dlnt[kc * sp..(kc + 1) * sp]
        .iter()
        .all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dx[kc * sp * sp..(kc + 1) * sp * sp]
        .iter()
        .all(|&v| v == 0.0));

    let expected_dxdt_h1 =
        (model.xa[kc * sp + i_h1] - xa_pre_step[kc * sp + i_h1]) / dt;
    assert!(
        (model.dxdt_nuc[kc * sp + i_h1] - expected_dxdt_h1).abs()
            <= 1.0e-15 * expected_dxdt_h1.abs().max(1.0e-30),
        "split-burn dxdt_nuc[h1]={} vs expected={expected_dxdt_h1}",
        model.dxdt_nuc[kc * sp + i_h1]
    );

    // Thermal neutrino losses must match fresh post-burn composition.
    let mut neu_ref = model.clone();
    set_basic_vars(&mut neu_ref).expect("set_basic_vars");
    do_eos(&mut neu_ref).expect("do_eos");
    do_neu(&mut neu_ref).expect("do_neu");
    assert_eq!(
        model.non_nuc_neu[kc].to_bits(),
        neu_ref.non_nuc_neu[kc].to_bits()
    );

    // Confirm coupled do_net would differ from split-burn dxdt_nuc & partials.
    let mut overwritten = model.clone();
    overwritten.aux.doing_struct_burn_mix = false;
    let net_info = resolve_model_net(&overwritten).expect("resolve_model_net");
    do_net(&mut overwritten, &net_info, false).expect("do_net");
    assert!(overwritten.d_epsnuc_dlnt[kc] != 0.0);

    // Non-split zones remain coupled.
    assert!(flexmesa_math::exp(model.ln_t_start[0]) < 1.0e7);
    assert_eq!(model.aux.burn_avg_epsnuc[0], 0.0);
}

#[test]
fn test_op_split_burn_zones_xa_start_infall_fold_and_do1_net_categories() {
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 1 Msun ZAMS");
    let sp = model.species;
    let kc = model.nz - 1;

    model.ln_t[kc] = flexmesa_math::log(1.8e8);
    model.ln_t_start = model.ln_t.clone();
    model.lnd[kc] = flexmesa_math::log(1.0e5);
    model.xa[kc * sp..kc * sp + sp]
        .copy_from_slice(&[0.25, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0]);
    set_basic_vars(&mut model).expect("set_basic_vars");
    do_eos(&mut model).expect("do_eos");

    // 1. All-positive v_start must NOT trip skip_burn via a 0.0 fold floor.
    model.controls.op_split_burn = true;
    model.controls.op_split_burn_min_t = 1.0e8;
    model.controls.op_split_burn_eps_nuc_infall_limit = -100.0;
    model.controls.eps_nuc_factor = 2.5;

    let xa_start_before = {
        model.save_start_values().expect("save_start_values");
        model.xa_start.clone()
    };
    model.v_start.fill(250.0);
    let xa_pre_burn = model.xa.clone();
    op_split_burn_zones(&mut model, 10.0).expect("op_split_burn_zones");

    assert!(model.aux.burn_avg_epsnuc[kc] > 0.0);
    assert_eq!(model.xa_start, xa_start_before);

    // 2. dxdt_nuc equals (ending xa - pre-burn xa) / dt while xa_start stays
    //    intact for caller inspection or retry rollback.
    for j in 0..sp {
        let expected =
            (model.xa[kc * sp + j] - xa_pre_burn[kc * sp + j]) / 10.0;
        assert!(
            (model.dxdt_nuc[kc * sp + j] - expected).abs() <= 1.0e-30,
            "species {j}: got={} expected={expected}",
            model.dxdt_nuc[kc * sp + j]
        );
    }

    // 3. Categories match do1_net on the post-burn xa with the pre-burn
    //    `abar`, `zbar`, `z2bar`, `ye` and `eta` that `burn1_zone`'s final
    //    `do1_net` reads (`struct_burn_mix.f90:1112`, `net.f90:260-266`).
    let mut net_ref = model.clone();
    let net_info = resolve_model_net(&net_ref).expect("resolve_model_net");
    do_net(&mut net_ref, &net_info, true).expect("do1_net categories");
    for c in 0..NUM_CATEGORIES {
        let got = model.aux.eps_nuc_categories[kc][c];
        let want = net_ref.aux.eps_nuc_categories[kc][c];
        assert!(
            (got - want).abs() <= 1.0e-12 * want.abs().max(1.0e-30),
            "category {c}: got={got} want={want}"
        );
    }
}

#[test]
fn test_do_net_zero_factors_clears_split_and_burn_avg_epsnuc_and_warnings() {
    let mut model = default_zams_approx21();
    let nz = model.nz;
    let sp = model.species;
    let kc = nz - 1;

    model.aux.doing_struct_burn_mix = true;
    model.controls.op_split_burn = true;
    model.controls.op_split_burn_min_t = 1.0e7;
    model.controls.eps_nuc_factor = 0.0;
    model.controls.dxdt_nuc_factor = 0.0;

    model.eps_nuc.fill(42.0);
    model.d_epsnuc_dlnd.fill(1.0);
    model.d_epsnuc_dlnt.fill(2.0);
    model.d_epsnuc_dx.fill(3.0);
    model.dxdt_nuc.fill(4.0);
    model.d_dxdt_nuc_dlnd.fill(5.0);
    model.d_dxdt_nuc_dlnt.fill(6.0);
    model.d_dxdt_nuc_dx.fill(7.0);
    model.eps_nuc_neu_total.fill(8.0);
    model.aux.burn_avg_epsnuc.fill(99.0);
    model.aux.eps_nuc_categories.fill([1.0; NUM_CATEGORIES]);

    let net_info = resolve_model_net(&model).expect("resolve_model_net");
    do_net(&mut model, &net_info, false).expect("do_net zero factors");

    assert!(model.eps_nuc.iter().all(|&v| v == 0.0));
    assert!(model.d_epsnuc_dlnd.iter().all(|&v| v == 0.0));
    assert!(model.d_epsnuc_dlnt.iter().all(|&v| v == 0.0));
    assert!(model.d_epsnuc_dx[..nz * sp].iter().all(|&v| v == 0.0));
    assert!(model.dxdt_nuc[..nz * sp].iter().all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dlnd[..nz * sp].iter().all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dlnt[..nz * sp].iter().all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dx[..nz * sp * sp]
        .iter()
        .all(|&v| v == 0.0));
    assert!(model.eps_nuc_neu_total.iter().all(|&v| v == 0.0));
    assert!(model.aux.burn_avg_epsnuc[..nz].iter().all(|&v| v == 0.0));
    assert!(model.aux.eps_nuc_categories[..nz]
        .iter()
        .all(|row| row.iter().all(|&v| v == 0.0)));

    // high_temp_rate_warning_zone gating: suppressed during doing_struct_burn_mix
    // and when abar > max_abar_for_burning (including max_abar == 0).
    model.ln_t[kc] = 10.5 * LN10;
    model.controls.output.warn_rates_for_high_temp = true;
    model.controls.output.max_safe_logt_for_rates = 10.0;
    assert_eq!(high_temp_rate_warning_zone(&model), None);

    model.aux.doing_struct_burn_mix = false;
    model.controls.max_abar_for_burning = 0.0;
    assert_eq!(high_temp_rate_warning_zone(&model), None);

    model.controls.max_abar_for_burning = 1.0e99;
    model.controls.max_logt_for_net = 10.2;
    let (warn_k, warn_logt) = high_temp_rate_warning_zone(&model)
        .expect("high-T rate warning must fire for core zone");
    assert_eq!(warn_k, kc);
    assert!((warn_logt - 10.2).abs() < 1.0e-14);
}

#[test]
fn test_change_net_sets_need_to_setvars_clears_derivs_and_local_net_cache() {
    let dir = TempDirGuard::new("micro_sbm_161_local_net");

    // 1. Overwriting a local .net file in search_dir must invalidate NET_CACHE.
    let net_path = dir.0.join("local_161.net");
    fs::write(&net_path, "add_isos(\n   h1\n   he4\n)\n")
        .expect("write 2-iso net");
    let n2 = get_or_load_net("local_161.net", Some(&dir.0))
        .expect("load 2-iso local net");
    assert_eq!(n2.num_isos, 2);

    fs::write(&net_path, "add_isos(\n   h1\n   he4\n   c12\n)\n")
        .expect("overwrite 3-iso net");
    let n3 = get_or_load_net("local_161.net", Some(&dir.0))
        .expect("reload 3-iso local net");
    assert_eq!(n3.num_isos, 3);

    // 2. change_model_network_with_adjust and extend_net set need_to_setvars
    //    and zero resized reaction/EOS composition derivative arrays.
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load ZAMS");
    set_hydro_vars(&mut model, false, false).expect("init hydro vars");
    model.aux.need_to_setvars = false;
    model.d_epsnuc_dx.fill(11.0);
    model.dxdt_nuc.fill(12.0);
    model.d_dxdt_nuc_dlnd.fill(13.0);
    model.d_dxdt_nuc_dlnt.fill(14.0);
    model.d_dxdt_nuc_dx.fill(15.0);
    model.aux.d_eps_grav_dx.fill(16.0);
    model.aux.dln_e_dxa_for_partials.fill(17.0);
    model.aux.dln_peos_dxa_for_partials.fill(18.0);

    change_model_network(&mut model, "co_burn.net", Some(&dir.0))
        .expect("switch to co_burn.net");
    assert!(model.aux.need_to_setvars);
    assert!(model.d_epsnuc_dx.iter().all(|&v| v == 0.0));
    assert!(model.dxdt_nuc.iter().all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dlnd.iter().all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dlnt.iter().all(|&v| v == 0.0));
    assert!(model.d_dxdt_nuc_dx.iter().all(|&v| v == 0.0));
    assert!(model.aux.d_eps_grav_dx.iter().all(|&v| v == 0.0));
    assert!(model.aux.dln_e_dxa_for_partials.iter().all(|&v| v == 0.0));
    assert!(model
        .aux
        .dln_peos_dxa_for_partials
        .iter()
        .all(|&v| v == 0.0));

    // Empty model guard in extend_net must not panic on nz == 0.
    let mut empty = model.clone();
    empty.nz = 0;
    assert!(!extend_net(&mut empty, &dir.0).expect("empty extend_net"));

    // extend_net transitioning co_net -> adv_net must also leave need_to_setvars set.
    model.aux.need_to_setvars = false;
    model.ln_t[model.nz - 1] = 9.1 * LN10;
    assert!(extend_net(&mut model, &dir.0).expect("extend_net to adv_net"));
    assert_eq!(model.job.net_name.trim(), model.job.adv_net.trim());
    assert!(model.aux.need_to_setvars);
}

#[test]
fn test_set_basic_vars_preserves_xa_and_micro_cache_tracks_change_net_and_logt()
{
    let mut model = load_zams_model(
        StarControls {
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load ZAMS");
    let sp = model.species;

    // 1. set_basic_vars must update composition moments without mutating xa
    //    so finite-difference composition Jacobian perturbations stay intact.
    let ref_bits: Vec<u64> =
        model.xa[0..sp].iter().map(|v| v.to_bits()).collect();
    model.xa[sp] += 1.0e-7;
    let perturbed_bit = model.xa[sp].to_bits();
    set_basic_vars(&mut model).expect("set_basic_vars");
    assert_eq!(model.xa[sp].to_bits(), perturbed_bit);
    for (j, &b) in ref_bits.iter().enumerate() {
        assert_eq!(model.xa[j].to_bits(), b);
    }

    // 2. MicrophysicsCache must invalidate on job.change_net/new_net_name and
    //    controls.output.max_safe_logt_for_rates.
    model.xa.copy_within(0..sp, sp);
    model.job.num_special_rate_factors = 1;
    model.job.reaction_for_special_factor = vec![String::from("rpp_to_he3")];
    model.job.special_rate_factor = vec![0.0];
    set_hydro_vars(&mut model, false, false).expect("populate cache");
    let misses_1 = model.micro_cache_misses;
    let eps_suppressed = model.eps_nuc[model.nz - 1];

    // Cache hit on unchanged state.
    set_hydro_vars(&mut model, false, false).expect("cache hit");
    assert_eq!(model.micro_cache_misses, misses_1);
    assert!(model.micro_cache_hits >= 1);

    // Pending change_net disables special_rate_factor in build_rate_factors.
    model.job.change_net = true;
    model.job.new_net_name = String::from("approx21.net");
    set_hydro_vars(&mut model, false, false)
        .expect("cache invalidates on change_net");
    assert_eq!(model.micro_cache_misses, misses_1 + 1);
    assert!(model.eps_nuc[model.nz - 1] > eps_suppressed);

    // Updating max_safe_logt_for_rates invalidates cached NetGeneralInfo.
    let misses_2 = model.micro_cache_misses;
    model.controls.output.max_safe_logt_for_rates = 9.5;
    set_hydro_vars(&mut model, false, false)
        .expect("cache invalidates on max_safe_logt_for_rates");
    assert_eq!(model.micro_cache_misses, misses_2 + 1);
}
