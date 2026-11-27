// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone unit and analytical-oracle tests for Issue `#157`/`#158`
//! covering convective premixing (`conv_premix`, `premix`), range microphysics
//! (`mixing_micro`), smoothing & composition gradients (`mixing_state`),
//! rotational mixing (`omega_mix`), cell energy accounting (`cell_energy`),
//! accretion composition (`accretion_xa`), mass-remap utilities (`mass_utils`),
//! and `StarModel` invariants (`model`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{L_SUN, M_SUN, R_SUN, SECYER, STANDARD_CGRAV, TWO_THIRDS};
use flexmesa_math::{exp, log, Quad};
use flexmesa_star::{
    build_accreted_xa, cell_specific_ke, cell_specific_pe,
    cell_specific_rotational_energy, cell_specific_total_energy, conv_premix,
    do_solve_omega_mix, get_dke_dt_dpe_dt, get_xa_for_accretion,
    load_zams_model, mass_utils, mixing_state, premix, set_hydro_vars,
    solve_omega_mix, total_angular_momentum, DkeDpeDt, OmegaMixOutcome,
    StarControls, StarError, StarJobControls, StarModel,
};
use flexmesa_turb::DrivingSpecies;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if actual == expected {
        return 0.0;
    }
    let denom = expected.abs().max(1.0e-99);
    (actual - expected).abs() / denom
}

fn iso_cid(name: &str) -> usize {
    NuclideCatalog::builtin_ref()
        .get_by_name(name)
        .expect("builtin nuclide must exist")
        .chem_id
}

fn find_species_index(model: &StarModel, z: usize, z_plus_n: usize) -> usize {
    let cat = NuclideCatalog::builtin_ref();
    model
        .chem_id
        .iter()
        .position(|&cid| {
            cat.get_by_id(cid)
                .is_some_and(|e| e.z == z && e.z_plus_n == z_plus_n)
        })
        .expect("nuclide present in network")
}

fn build_cpm_ledoux_model() -> StarModel {
    let ctrl = StarControls {
        initial_mass: 4.0,
        initial_z: 0.02,
        zbase: 0.02,
        mixing_length_alpha: 1.8,
        use_ledoux_criterion: true,
        do_conv_premix: true,
        ..StarControls::default()
    };
    let mut model = load_zams_model(ctrl, StarJobControls::default())
        .expect("load 4 Msun ZAMS");
    model.dt = 3.418e6 * SECYER;
    model.controls.mix_wind.conv_premix_avoid_increase = true;
    model.controls.mix_wind.conv_premix_time_factor = 1.0;
    model.controls.mix_wind.conv_premix_fix_pgas = true;
    model.aux.mixing.okay_to_set_mixing_info = true;
    set_hydro_vars(&mut model, false, false).expect("initial set_hydro_vars");
    model.sync_mesh_and_pack().expect("sync_mesh_and_pack");

    let sp = model.species;
    let i_h1 = find_species_index(&model, 1, 1);
    let i_he4 = find_species_index(&model, 2, 4);
    let k_bdy = model.aux.mixing.conv_bdy_loc[0];
    let k_grad_top = (k_bdy + 30).min(model.nz - 25);
    let k_grad_bot = (k_grad_top + 12).min(model.nz - 10);
    let span = f64::from(
        u32::try_from((k_grad_bot - k_grad_top).max(1)).expect("small span"),
    );
    for k in k_grad_top..model.nz {
        let frac = if k < k_grad_bot {
            f64::from(u32::try_from(k - k_grad_top).expect("small offset"))
                / span
        } else {
            1.0
        };
        let delta = (0.015 * frac).min(model.xa[k * sp + i_h1]);
        model.xa[k * sp + i_h1] -= delta;
        model.xa[k * sp + i_he4] += delta;
    }
    model.aux.mixing.okay_to_set_mixing_info = true;
    set_hydro_vars(&mut model, false, false).expect("refresh hydro_vars");
    model.sync_mesh_and_pack().expect("sync_mesh_and_pack");
    model
}

#[test]
fn test_star_model_new_invariants_and_xmstar_old_consistency() {
    let nz = 6_usize;
    let chem_id = vec![iso_cid("h1"), iso_cid("he4")];
    let ctrl = StarControls {
        initial_mass: 2.5,
        initial_z: 0.014,
        zbase: 0.014,
        ..StarControls::default()
    };
    let mut model =
        StarModel::new(nz, chem_id.clone(), StarJobControls::default(), ctrl);

    let expected_mstar = 2.5 * M_SUN;
    assert_eq!(model.nz, nz);
    assert_eq!(model.species, 2);
    assert_eq!(model.mstar, expected_mstar);
    assert_eq!(model.mstar_start, expected_mstar);
    assert_eq!(model.mstar_old, expected_mstar);
    assert_eq!(model.xmstar, expected_mstar);
    assert_eq!(model.m_center, 0.0);
    assert_eq!(model.m_center_old, 0.0);
    // `mstar_old == m_center_old + xmstar_old` invariant matches
    // `StarModel::new`'s pre-step defaults
    // (`star_data/public/star_data_step_input.inc:26-28`,
    // `star/private/evolve_support.f90:84-86, 157-159`).
    assert_eq!(model.xmstar_old, expected_mstar);
    assert_eq!(
        model.mstar_old,
        model.m_center_old + model.xmstar_old,
        "mstar_old must equal m_center_old + xmstar_old on StarModel::new"
    );
    assert!(model.doing_first_model_of_run);
    assert_eq!(model.iso_names, vec!["h1", "he4"]);
    assert_eq!(model.net_iso[chem_id[0]], Some(0));
    assert_eq!(model.net_iso[chem_id[1]], Some(1));
    assert_eq!(model.kap_config.zbase, 0.014);
    assert_eq!(model.tau_base, 2.0 / 3.0);

    model.cell_xa_mut(2).copy_from_slice(&[0.71, 0.29]);
    assert_eq!(model.cell_xa(2), &[0.71, 0.29]);

    model.lum[0] = 3.2 * L_SUN;
    model.ln_r[0] = log(2.5 * R_SUN);
    assert!(rel_err(model.surface_l_lsun(), 3.2) < 1.0e-15);
    assert!(rel_err(model.surface_r_rsun(), 2.5) < 1.0e-14);
    assert!(model.heap_bytes() > 0);
}

#[test]
fn test_accretion_xa_given_mass_fractions_nan_and_h2_without_h1() {
    let h1_id = iso_cid("h1");
    let h2_id = iso_cid("h2");
    let he4_id = iso_cid("he4");

    // 1. accrete_same_as_surface + accrete_given_mass_fractions normalization & NaN.
    let mut model = StarModel::new(
        2,
        vec![h1_id, he4_id],
        StarJobControls::default(),
        StarControls::default(),
    );
    model.cell_xa_mut(0).copy_from_slice(&[0.64, 0.36]);
    model.controls.accrete_same_as_surface = true;
    assert_eq!(
        get_xa_for_accretion(&model).expect("same as surface"),
        vec![0.64, 0.36]
    );

    model.controls.accrete_same_as_surface = false;
    model.controls.accrete_given_mass_fractions = true;
    model.controls.mix_wind.num_accretion_species = 2;
    model.controls.accretion_species_id =
        vec![String::from("h1"), String::from("he4")];
    model.controls.accretion_species_xa = vec![0.7035, 0.3015]; // sum = 1.005
    let norm = get_xa_for_accretion(&model).expect("normalized within tol");
    assert!(rel_err(norm[0], 0.7) < 1.0e-14);
    assert!(rel_err(norm[1], 0.3) < 1.0e-14);

    model.controls.accretion_species_xa = vec![f64::NAN, 0.25];
    assert!(
        matches!(
            get_xa_for_accretion(&model),
            Err(StarError::InvalidInput(_))
        ),
        "NaN accretion_species_xa must fail loudly"
    );
    assert_eq!(build_accreted_xa(&model), vec![0.64, 0.36]);

    // 2. Net containing h2 without h1 must error unconditionally (even when
    //    `accretion_h1 == 0.0`), because upstream writes
    //    `xa(net_iso(ih1)) = h1` out of bounds at `xa(0)`
    //    (`star/private/adjust_xyz.f90:1134-1136`; DECISIONS #43, #564).
    let mut h2_only = StarModel::new(
        2,
        vec![h2_id, he4_id],
        StarJobControls::default(),
        StarControls {
            accrete_same_as_surface: false,
            accretion_h1: 0.70,
            accretion_he4: 0.28,
            ..StarControls::default()
        },
    );
    h2_only.controls.mix_wind.accretion_h2 = 0.02;
    h2_only.controls.mix_wind.accretion_he3 = 0.0;
    h2_only.controls.mix_wind.accretion_zfracs = 3;
    match get_xa_for_accretion(&h2_only) {
        Err(StarError::InvalidInput(msg)) => {
            assert_ne!(msg, "get_xa: require h1 to be in net");
            assert!(msg.contains("#564"), "unexpected error message: {msg}");
        }
        other => {
            panic!("expected InvalidInput for h2 without h1, got {other:?}")
        }
    }

    h2_only.controls.accretion_h1 = 0.0;
    h2_only.controls.mix_wind.accretion_h2 = 0.72;
    match get_xa_for_accretion(&h2_only) {
        Err(StarError::InvalidInput(msg)) => {
            assert_ne!(msg, "get_xa: require h1 to be in net");
            assert!(msg.contains("#564"), "unexpected error message: {msg}");
        }
        other => panic!(
            "expected unconditional InvalidInput for h2 without h1 at h1=0, got {other:?}"
        ),
    }

    // 3. When h2 is absent and h1 is present, h2 folds into h1
    //    (`xa[h1] = h1 + h2`). Also exercises upstream
    //    `star/private/adjust_xyz.f90:1172-1209` `jskip` on H+He nets.
    let mut no_h2 = StarModel::new(
        2,
        vec![h1_id, he4_id],
        StarJobControls::default(),
        StarControls {
            accrete_same_as_surface: false,
            accretion_h1: 0.70,
            accretion_he4: 0.28,
            ..StarControls::default()
        },
    );
    no_h2.controls.mix_wind.accretion_h2 = 0.02;
    no_h2.controls.mix_wind.accretion_he3 = 0.0;
    no_h2.controls.mix_wind.accretion_zfracs = 3;
    no_h2
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = false;
    let xa_folded = get_xa_for_accretion(&no_h2).expect("metal-free h1+h2");
    assert!(rel_err(xa_folded[0], 0.72) < 1.0e-14);
    assert!(rel_err(xa_folded[1], 0.28) < 1.0e-14);

    no_h2
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = true;
    let xa_upstream_quirk =
        get_xa_for_accretion(&no_h2).expect("upstream jskip quirk on H+He");
    assert!(rel_err(xa_upstream_quirk[0], 1.0) < 1.0e-14);
    assert_eq!(xa_upstream_quirk[1], 0.0);
}

#[test]
fn test_accretion_xa_standard_metals_zfracs_and_heaviest_dump() {
    let chem_id = vec![
        iso_cid("h1"),
        iso_cid("h2"),
        iso_cid("he3"),
        iso_cid("he4"),
        iso_cid("c12"),
        iso_cid("n14"),
        iso_cid("o16"),
        iso_cid("ne20"),
        iso_cid("fe56"),
    ];
    let mut model = StarModel::new(
        2,
        chem_id,
        StarJobControls::default(),
        StarControls {
            accrete_same_as_surface: false,
            accretion_h1: 0.70,
            accretion_he4: 0.2799,
            ..StarControls::default()
        },
    );
    model.controls.mix_wind.accretion_h2 = 5.0e-5;
    model.controls.mix_wind.accretion_he3 = 5.0e-5;

    for zf in 1..=10 {
        model.controls.mix_wind.accretion_zfracs = zf;
        let xa = get_xa_for_accretion(&model).expect("valid zfracs 1..=10");
        assert!((xa.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
        assert!(rel_err(xa[0], 0.70) < 1.0e-14);
        assert!(rel_err(xa[1], 5.0e-5) < 1.0e-14);
        assert!(rel_err(xa[2], 5.0e-5) < 1.0e-14);
        assert!(rel_err(xa[3], 0.2799) < 1.0e-14);
    }

    // Invalid zfracs is rejected when Z > 1e-8, but ignored when metal-free.
    model.controls.mix_wind.accretion_zfracs = 99;
    assert!(matches!(
        get_xa_for_accretion(&model),
        Err(StarError::InvalidInput(_))
    ));
    let mut mf = model.clone();
    mf.controls.accretion_h1 = 0.72;
    mf.controls.accretion_he4 = 0.28;
    mf.controls.mix_wind.accretion_h2 = 0.0;
    mf.controls.mix_wind.accretion_he3 = 0.0;
    let xa_mf = get_xa_for_accretion(&mf).expect("metal-free allows zfracs=99");
    assert!(rel_err(xa_mf[0], 0.72) < 1.0e-14);

    // `dump_missing_metals_into_heaviest` keeps lighter metals unscaled and
    // puts missing elements into `fe56`.
    model.controls.mix_wind.accretion_zfracs = 3;
    model
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = false;
    let xa_renorm = get_xa_for_accretion(&model).expect("renorm");
    model
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = true;
    let xa_heaviest = get_xa_for_accretion(&model).expect("dump heaviest");
    assert!(
        xa_heaviest[8] > xa_renorm[8],
        "fe56 must absorb missing metals"
    );
    assert!(
        xa_heaviest[4] < xa_renorm[4],
        "c12 is not boosted when dumped"
    );

    // Negative custom z_fraction fails loudly.
    model.controls.mix_wind.accretion_zfracs = 0;
    model.controls.mix_wind.z_fraction_c = -0.1;
    model.controls.mix_wind.z_fraction_o = 1.1;
    assert!(matches!(
        get_xa_for_accretion(&model),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn test_conv_premix_updates_mass_correction_and_premix_early_return_flag() {
    // 1. Degenerate `premix::do_conv_premix` (`nz < 2` and `species == 0`) must
    //    set `need_to_setvars = true` matching `star/private/conv_premix.f90:176`.
    let mut one_zone = StarModel::new(
        1,
        vec![iso_cid("h1"), iso_cid("he4")],
        StarJobControls::default(),
        StarControls::default(),
    );
    one_zone.aux.need_to_setvars = false;
    assert_eq!(
        premix::do_conv_premix(&mut one_zone, 10, true).expect("ok"),
        0
    );
    assert!(one_zone.aux.need_to_setvars);

    let mut no_species = StarModel::new(
        4,
        Vec::new(),
        StarJobControls::default(),
        StarControls::default(),
    );
    no_species.aux.need_to_setvars = false;
    assert_eq!(
        premix::do_conv_premix(&mut no_species, 10, true).expect("ok"),
        0
    );
    assert!(no_species.aux.need_to_setvars);

    // 2. Idle conv_premix leaves `mass_correction` bit-identical (`CellRangeSnapshot`
    //    zero drift) while setting `need_to_setvars`.
    let mut idle = build_cpm_ledoux_model();
    idle.controls.mix_wind.conv_premix_time_factor = 1.0e-20;
    let mc_idle_before: Vec<u64> = idle
        .aux
        .mass_correction
        .iter()
        .map(|v| v.to_bits())
        .collect();
    idle.aux.need_to_setvars = false;
    let adv_idle = conv_premix::do_conv_premix(&mut idle).expect("idle cpm");
    assert_eq!(adv_idle, 0);
    assert!(idle.aux.need_to_setvars);
    let mc_idle_after: Vec<u64> = idle
        .aux
        .mass_correction
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(mc_idle_before, mc_idle_after);

    // 3. Active conv_premix advancing cells updates `model.aux.mass_correction[k]`
    //    from `basic_composition_info` (`star/private/micro.f90:321-324`).
    let mut active = build_cpm_ledoux_model();
    let xa_before = active.xa.clone();
    active.aux.mass_correction.fill(0.5); // sentinel to detect updated cells
    let adv = conv_premix::do_conv_premix(&mut active).expect("active cpm");
    assert!(adv > 0, "expected advancing premix zones");

    let cat = NuclideCatalog::builtin_ref();
    let mut checked = 0_usize;
    for k in 0..active.nz {
        let xa_k = active.cell_xa(k);
        let sp = active.species;
        if xa_k != &xa_before[k * sp..(k + 1) * sp] {
            let comp = get_composition_info(cat, &active.chem_id, xa_k, true)
                .expect("comp");
            assert!(
                rel_err(active.aux.mass_correction[k], comp.mass_correction)
                    < 1.0e-14,
                "mass_correction at mixed cell {k} must match \
                 get_composition_info"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "expected modified composition cells");
}

#[test]
fn test_weighted_threshold_smoothing_and_mixing_state_oracles() {
    // 1. Exact `[1, 2, 1] / 4` binomial stencil (`ns = 1`, `preserve_sign = false`).
    let mut dd = [1.0, 2.0, 10.0, 4.0, 5.0];
    let mut work = [0.0; 5];
    mixing_state::weighted_smoothing(&mut dd, 0, false, &mut work);
    assert_eq!(dd, [1.0, 2.0, 10.0, 4.0, 5.0]);

    mixing_state::weighted_smoothing(&mut dd, 1, false, &mut work);
    assert_eq!(dd[0], 1.0);
    assert_eq!(dd[4], 5.0);
    assert!(rel_err(dd[1], 15.0 / 4.0) < 1.0e-15);
    assert!(rel_err(dd[2], 26.0 / 4.0) < 1.0e-15);
    assert!(rel_err(dd[3], 23.0 / 4.0) < 1.0e-15);

    // 2. `preserve_sign = true` stops stencil sum on sign reversal.
    let mut dd_sign = [1.0, 2.0, -4.0, 8.0, 10.0];
    mixing_state::weighted_smoothing(&mut dd_sign, 1, true, &mut work);
    assert_eq!(dd_sign[0], 1.0);
    assert!(rel_err(dd_sign[1], 5.0 / 3.0) < 1.0e-15);
    assert_eq!(dd_sign[2], -4.0);
    assert!(rel_err(dd_sign[3], 26.0 / 3.0) < 1.0e-15);
    assert_eq!(dd_sign[4], 10.0);

    // 3. `threshold_smoothing` only smooths interiors of `|dd| >= thresh` spans.
    let mut dd_th = [0.1, 1.0, 2.0, 10.0, 4.0, 5.0, 0.2];
    let mut work_th = [0.0; 7];
    mixing_state::threshold_smoothing(&mut dd_th, 0.5, 1, false, &mut work_th);
    assert_eq!(dd_th[0], 0.1);
    assert_eq!(dd_th[1], 1.0);
    assert_eq!(dd_th[5], 5.0);
    assert_eq!(dd_th[6], 0.2);
    assert!(rel_err(dd_th[2], 15.0 / 4.0) < 1.0e-15);

    // 4. `driving_species_for_cid` exact (Z, A) mapping and fallback.
    let ds_he3 = mixing_state::driving_species_for_cid(iso_cid("he3"));
    assert_eq!(ds_he3, DrivingSpecies::he3());
    let ds_c12 = mixing_state::driving_species_for_cid(iso_cid("c12"));
    assert_eq!(ds_c12, DrivingSpecies::c12());
    assert_eq!(
        mixing_state::driving_species_for_cid(0),
        DrivingSpecies::he4()
    );

    // 5. `set_grads` thermohaline `dominant_iso_for_thermohaline` selection.
    let nz = 12_usize;
    let h1_id = iso_cid("h1");
    let he3_id = iso_cid("he3");
    let he4_id = iso_cid("he4");
    let mut model = StarModel::new(
        nz,
        vec![h1_id, he3_id, he4_id],
        StarJobControls::default(),
        StarControls {
            use_ledoux_criterion: true,
            calculate_brunt_b: true,
            thermohaline_coeff: 2.0,
            ..StarControls::default()
        },
    );
    for k in 0..nz {
        model.ln_peos[k] = 30.0 + 0.2 * f64::from(u32::try_from(k).expect("k"));
        model.lnd[k] = 1.0 + 0.1 * f64::from(u32::try_from(k).expect("k"));
        model.ln_t[k] = 15.0 + 0.05 * f64::from(u32::try_from(k).expect("k"));
        let x_he3 = 0.28 - 0.02 * f64::from(u32::try_from(k).expect("k"));
        let x_he4 = 0.02 * f64::from(u32::try_from(k).expect("k"));
        model.cell_xa_mut(k).copy_from_slice(&[0.72, x_he3, x_he4]);
    }
    mixing_state::set_grads(&mut model);
    assert!(
        model.aux.mixing.dominant_iso_for_thermohaline[4..8]
            .iter()
            .all(|&cid| cid == he3_id),
        "he3 gradient must win over equal-magnitude he4 gradient via (1+Z)/A"
    );
}

#[test]
fn test_omega_mix_standalone_irot_fallback_and_conservation() {
    // 1. Uninitialized/zero `i_rot` falls back to `2/3 r^2`, updates `j_rot`,
    //    and conserves total angular momentum on a 4 Msun ZAMS profile.
    let mut zams = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("load 4 Msun ZAMS");
    let nz = zams.nz;
    for (k, om) in zams.omega.iter_mut().enumerate().take(nz) {
        let u = f64::from(u32::try_from(k).expect("k"))
            / f64::from(u32::try_from(nz - 1).expect("nz-1"));
        *om = 1.0e-5 * (1.0 + 2.0 * u);
    }
    zams.aux.i_rot.fill(0.0);
    let expected_j0: f64 = (0..nz)
        .map(|k| {
            let i_rot = TWO_THIRDS * zams.r[k] * zams.r[k];
            zams.dm_bar[k] * i_rot * zams.omega[k]
        })
        .sum();
    let nu = vec![1.0e13; nz];
    solve_omega_mix(&mut zams, 1.0e6, &nu, 4);

    for k in 0..nz {
        assert!(zams.aux.i_rot[k] > 0.0);
        assert!(
            rel_err(zams.j_rot[k], zams.aux.i_rot[k] * zams.omega[k]) < 1.0e-15
        );
    }
    let j1: f64 = (0..nz).map(|k| zams.dm_bar[k] * zams.j_rot[k]).sum();
    assert!(
        rel_err(j1, expected_j0) < 1.0e-12,
        "standalone solve_omega_mix must conserve J_tot"
    );

    // 2. Uniform omega with zero torque stays stationary.
    let mut flat = load_zams_model(
        StarControls {
            initial_mass: 4.0,
            rotation_flag: true,
            zbase: 0.02,
            ..StarControls::default()
        },
        StarJobControls::default(),
    )
    .expect("4 Msun rot ZAMS");
    flat.omega.fill(3.0e-5);
    let nu_flat = vec![1.0e14; flat.nz];
    solve_omega_mix(&mut flat, 1.0e7, &nu_flat, 4);
    for (k, &om) in flat.omega.iter().enumerate() {
        assert!(
            rel_err(om, 3.0e-5) < 1.0e-13,
            "uniform omega must remain stationary at k={k}, got {om}"
        );
    }

    // 3. `do_solve_omega_mix` with `dt_total <= 0.0` updates `omega = j_rot / i_rot`
    //    via `set_i_rot` + `set_omega` and returns `KeepGoing`.
    let mut rot_dt0 = flat.clone();
    rot_dt0.controls.rotation_flag = true;
    rot_dt0.j_rot.fill(6.0e15);
    assert_eq!(
        do_solve_omega_mix(&mut rot_dt0, 0.0).expect("dt <= 0"),
        OmegaMixOutcome::KeepGoing
    );
    assert!(rot_dt0.aux.i_rot[0] > 0.0);
    assert!(
        rel_err(rot_dt0.omega[0], rot_dt0.j_rot[0] / rot_dt0.aux.i_rot[0])
            < 1.0e-14
    );
    assert!(total_angular_momentum(&rot_dt0) > 0.0);

    // 4. `do_solve_omega_mix` on `nz == 1` (`nz < 2`) updates `omega[0]` and returns
    //    `KeepGoing` without diffusion.
    let mut one_zone = StarModel::new(
        1,
        vec![iso_cid("h1")],
        StarJobControls::default(),
        StarControls {
            rotation_flag: true,
            ..StarControls::default()
        },
    );
    one_zone.r = vec![R_SUN];
    one_zone.ln_r = vec![log(R_SUN)];
    one_zone.j_rot = vec![4.0e15];
    assert_eq!(
        do_solve_omega_mix(&mut one_zone, 1.0e5).expect("nz == 1"),
        OmegaMixOutcome::KeepGoing
    );
    assert!(
        rel_err(one_zone.omega[0], one_zone.j_rot[0] / one_zone.aux.i_rot[0])
            < 1.0e-14
    );
}

#[test]
fn test_cell_energy_pe_ke_rot_total_and_dke_dt_dpe_dt_oracles() {
    let nz = 3_usize;
    let mut m = StarModel::new(
        nz,
        vec![iso_cid("h1")],
        StarJobControls::default(),
        StarControls {
            v_flag: true,
            u_flag: false,
            rotation_flag: true,
            include_rotation_in_total_energy: true,
            use_mass_corrections: true,
            rsp_flag: true,
            ..StarControls::default()
        },
    );
    m.cgrav = vec![STANDARD_CGRAV; nz];
    m.m_grav = vec![3.0e33, 2.0e33, 1.0e33];
    m.aux.mass_correction = vec![1.004, 1.002, 1.001];
    m.r = vec![4.0e10, 2.5e10, 1.0e10];
    m.ln_r = m.r.iter().map(|&r| log(r)).collect();
    m.ln_r_start = vec![log(4.2e10), log(2.6e10), log(1.1e10)];
    m.v = vec![3.0e5, 2.0e5, 1.0e5];
    m.v_start = vec![2.5e5, 1.5e5, 0.5e5];
    m.v_center = 1.0e4;
    m.energy = vec![2.0e14, 3.0e14, 4.0e14];
    m.omega = vec![1.0e-4, 2.0e-4, 3.0e-4];
    m.aux.i_rot = vec![1.0e20, 2.0e20, 3.0e20];
    m.aux.rsp.et_rsp = vec![5.0e11, 6.0e11, 7.0e11];

    // Closed-form PE with `mass_correction`
    // (`star/private/star_utils.f90:2192-2196,2256-2266`).
    let pe0_expected = -0.5
        * 1.004
        * (STANDARD_CGRAV * 3.0e33 / 4.0e10 + STANDARD_CGRAV * 2.0e33 / 2.5e10);
    let pe_last_expected = -0.5 * (STANDARD_CGRAV * 1.0e33 * 1.001 / 1.0e10);
    assert!(rel_err(cell_specific_pe(&m, 0), pe0_expected) < 1.0e-14);
    assert!(rel_err(cell_specific_pe(&m, nz - 1), pe_last_expected) < 1.0e-14);

    // Closed-form KE (`v_flag`, including `mass_correction` and innermost `v_center`).
    let ke0_expected = 0.25 * 1.004 * (3.0e5 * 3.0e5 + 2.0e5 * 2.0e5);
    let ke_last_expected = 0.25 * 1.001 * (1.0e5 * 1.0e5 + 1.0e4 * 1.0e4);
    assert!(rel_err(cell_specific_ke(&m, 0), ke0_expected) < 1.0e-15);
    assert!(rel_err(cell_specific_ke(&m, nz - 1), ke_last_expected) < 1.0e-15);

    // Rotational & total energy.
    let rot0 = 0.5 * (1.0e20 * 1.0e-8 + 2.0e20 * 4.0e-8);
    assert!(rel_err(cell_specific_rotational_energy(&m, 0), rot0) < 1.0e-15);
    let tot0 = m.energy[0] + ke0_expected + pe0_expected + rot0 + 5.0e11;
    assert!(rel_err(cell_specific_total_energy(&m, 0), tot0) < 1.0e-14);

    // `get_dke_dt_dpe_dt`: exact finite-difference rate + analytical derivatives.
    let dt = 100.0;
    let rates: DkeDpeDt = get_dke_dt_dpe_dt(&m, 0, dt);
    let pe0_start = -0.5
        * 1.004
        * (STANDARD_CGRAV * 3.0e33 / exp(m.ln_r_start[0])
            + STANDARD_CGRAV * 2.0e33 / exp(m.ln_r_start[1]));
    let ke0_start = 0.25 * 1.004 * (2.5e5 * 2.5e5 + 1.5e5 * 1.5e5);
    assert!(rel_err(rates.dpe_dt, (pe0_expected - pe0_start) / dt) < 1.0e-13);
    assert!(rel_err(rates.dke_dt, (ke0_expected - ke0_start) / dt) < 1.0e-14);
    assert!(
        rel_err(
            rates.d_dpedt_dlnr00,
            (0.5 * 1.004 * STANDARD_CGRAV * 3.0e33 / 4.0e10) / dt
        ) < 1.0e-14
    );
    assert!(
        rel_err(
            rates.d_dpedt_dlnrp1,
            (0.5 * 1.004 * STANDARD_CGRAV * 2.0e33 / 2.5e10) / dt
        ) < 1.0e-14
    );
    assert!(rel_err(rates.d_dkedt_dv00, 0.5 * 1.004 * 3.0e5 / dt) < 1.0e-15);
    assert!(rel_err(rates.d_dkedt_dvp1, 0.5 * 1.004 * 2.0e5 / dt) < 1.0e-15);

    // Innermost cell has `d_dkedt_dvp1 = 0` (`v_center` is fixed).
    let rates_last = get_dke_dt_dpe_dt(&m, nz - 1, dt);
    assert_eq!(rates_last.d_dkedt_dvp1, 0.0);
    assert_eq!(rates_last.d_dpedt_dlnrp1, 0.0);

    // Inner face `rp1 <= 0` guard sets `gravp1 = 0` on cell `nz - 2`, while
    // `r00 <= 0` on cell `nz - 1` matches upstream (`-inf`, no `r00` guard).
    m.r[nz - 1] = 0.0;
    let pe_mid_expected = -0.5 * 1.002 * (STANDARD_CGRAV * 2.0e33 / 2.5e10);
    assert!(rel_err(cell_specific_pe(&m, nz - 2), pe_mid_expected) < 1.0e-14);
    assert!(cell_specific_pe(&m, nz - 1).is_infinite());
}

#[test]
fn test_mass_utils_quad_difference_and_conserved_remap_oracles() {
    // 1. `accurate_mass_difference` across 15 orders of magnitude in Quad precision.
    let nz = 3_usize;
    let dm1 = mass_utils::one_based(&[1.0e18, 2.0e18, 1.0e33]);
    let dm2 = mass_utils::one_based(&[2.5e18, 2.0e18, 1.0e33]);
    // Sum from j=1 (`m_old = 1e33 + 3e18`) vs k=1 (`m_new = 1e33 + 4.5e18`).
    let diff = mass_utils::accurate_mass_difference(&dm1, &dm2, 1, 1, nz);
    assert!(
        rel_err(diff.to_f64(), -1.5e18) < 1.0e-14,
        "Quad mass difference must resolve 1.5e18 on top of 1e33, got {}",
        diff.to_f64()
    );

    let m_surf = mass_utils::reconstruct_m(&dm1, nz, 1);
    let xm_all = mass_utils::reconstruct_xm(&dm1, nz);
    assert_eq!(m_surf, xm_all);

    // 2. `integrate_conserved` under accretion spanning > 1 new cell
    //    (`sum(dm_new) > sum(dm_old)`).
    let dm_old = mass_utils::one_based(&[1.0, 2.0, 3.0]); // sum = 6.0
    let dm_gain = mass_utils::one_based(&[1.5, 2.0, 4.5]); // sum = 8.0 (+2.0 accreted)
    let vals_old = vec![vec![0.0], vec![10.0], vec![20.0], vec![30.0]];
    let vals_outside = [50.0];
    let vals_gain = mass_utils::integrate_conserved(
        &vals_old,
        &vals_outside,
        &dm_gain,
        &dm_old,
        nz,
        1,
    );
    // Exact integrals from outside inward:
    // Cell 1 (`dm=1.5`): 1.5*50 -> mean 50.0
    // Cell 2 (`dm=2.0`): 0.5*50 + 1.0*10 + 0.5*20 = 45 -> mean 22.5
    // Cell 3 (`dm=4.5`): 1.5*20 + 3.0*30 = 120 -> mean 80/3
    assert!(rel_err(vals_gain[1][0], 50.0) < 1.0e-14);
    assert!(rel_err(vals_gain[2][0], 22.5) < 1.0e-14);
    assert!(rel_err(vals_gain[3][0], 80.0 / 3.0) < 1.0e-14);

    // Verify pass fractions & `find_j_ranges` consistency on the same gain mesh.
    let ci = mass_utils::make_compressed_intersect(&dm_gain, &dm_old, nz);
    let (i_min, i_max) = mass_utils::find_i_ranges(nz, &ci.ranges);
    assert!(i_min[1] <= i_max[1]);
    let pf = mass_utils::prepare_pass_fraction(
        nz,
        Quad::from_f64(2.0),
        &dm_gain,
        &ci.mesh_intersects,
        &ci.ranges,
    );
    let _p11 = mass_utils::compute_pass_fraction(1, 1, &pf);
    let mass_flux = vec![Quad::ZERO; nz + 2];
    let (j_min, j_max) = mass_utils::find_j_ranges(nz, &ci.ranges, &mass_flux);
    assert_eq!(j_min.len(), nz + 1);
    assert_eq!(j_max.len(), nz + 1);

    // 3. `integrate_conserved` under mass loss (`sum(dm_new) < sum(dm_old)`).
    let dm_loss = mass_utils::one_based(&[1.5, 2.0, 1.5]); // sum = 5.0 (-1.0 lost)
    let vals_loss = mass_utils::integrate_conserved(
        &vals_old,
        &vals_outside,
        &dm_loss,
        &dm_old,
        nz,
        1,
    );
    // Removed 1.0 from top (`cell 1` of old, val=10) leaving 2.0 in `cell 2` + 3.0 in `cell 3`:
    // New cell 1 (`dm=1.5`): 1.5*20 -> 20.0
    // New cell 2 (`dm=2.0`): 0.5*20 + 1.5*30 = 55 -> 27.5
    // New cell 3 (`dm=1.5`): 1.5*30 -> 30.0
    assert!(rel_err(vals_loss[1][0], 20.0) < 1.0e-14);
    assert!(rel_err(vals_loss[2][0], 27.5) < 1.0e-14);
    assert!(rel_err(vals_loss[3][0], 30.0) < 1.0e-14);
}
