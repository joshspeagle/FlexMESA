// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone Phase A parity tests for MESA `26.4.1` mass adjustment,
//! accreted composition, `eps_mdot`, `mass_utils`, and critical-rotation
//! `mdot_redo` (`#153`, `#161`).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::manual_midpoint,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]

use flexmesa_chem::{
    lodders03_element_atom_percent, solar_element_zfrac, NuclideCatalog,
    SolarScale,
};
use flexmesa_const::{FOUR_THIRDS_PI, M_SUN, SECYER};
use flexmesa_math::Quad;
use flexmesa_star::accretion_xa::build_accreted_xa;
use flexmesa_star::hydro_rotation::{
    set_rotation_info, use_xh_to_update_i_rot_and_j_rot,
};
use flexmesa_star::mass_utils::{
    accurate_mass_difference, compute_pass_fraction, find_i_ranges,
    find_j_ranges, integrate_conserved, make_compressed_intersect, one_based,
    prepare_pass_fraction, reconstruct_m, reconstruct_xm, zero_slot,
};
use flexmesa_star::report_utils::{
    equatorial_and_polar_r_cm, eval_upstream_omega_crit, get_lrad_div_ledd,
};
use flexmesa_star::{
    calculate_eps_mdot, compute_delta_m, do_adjust_mass, do_adjust_mass_step,
    eval_surf_avg_rotation_info, eval_surf_avg_w_div_w_crit,
    eval_total_energy_profile, find_mass_flux, get_xa_for_accretion,
    load_zams_model, set_hydro_vars, total_angular_momentum,
    total_energy_profile_before_adjust_mass, AdjustMassRecord, MdotRedoState,
    SelectMdotAction, StarControls, StarError, StarExtras, StarJobControls,
    StarModel,
};

fn zams_model(mass_msun: f64) -> StarModel {
    let ctrl = StarControls {
        initial_mass: mass_msun,
        zbase: 0.02,
        initial_z: 0.02,
        ..StarControls::default()
    };
    let mut model =
        load_zams_model(ctrl, StarJobControls::default()).expect("load zams");
    set_hydro_vars(&mut model, false, false).expect("set_hydro_vars");
    model.save_old_values();
    model
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    let denom = expected.abs().max(1.0e-30);
    (actual - expected).abs() / denom
}

fn dm_bar_from_dm(dm: &[f64]) -> Vec<f64> {
    let nz = dm.len();
    let mut out = vec![0.0_f64; nz];
    if nz == 0 {
        return out;
    }
    out[0] = 0.5 * dm[0];
    for k in 1..nz {
        out[k] = 0.5 * (dm[k - 1] + dm[k]);
    }
    out[nz - 1] = 0.5 * dm[nz - 2] + dm[nz - 1];
    out
}

struct FixedWind(f64);

impl StarExtras for FixedWind {
    fn other_wind(
        &self,
        _m: &StarModel,
        _l: f64,
        _ms: f64,
        _r: f64,
        _t: f64,
    ) -> Result<Option<f64>, StarError> {
        Ok(Some(self.0))
    }
}

#[test]
fn mass_utils_quad_difference_pass_fractions_and_integrate_conserved_parity() {
    let dm1 = one_based(&[1.0_f64, 0.25, 1.0e-15, 0.5e-15]);
    let dm2 = one_based(&[1.0_f64, 0.25, 2.0e-15, 0.5e-15]);

    assert_eq!(reconstruct_xm(&dm1, 2), Quad::from(1.25));
    let m_inner = reconstruct_m(&dm1, 4, 3);
    assert!((m_inner.to_f64() - 1.5e-15).abs() < 1.0e-30);

    // `accurate_mass_difference` must preserve deep 1e-15 cell differences
    // beneath an O(1) outer envelope.
    let diff = accurate_mass_difference(&dm1, &dm2, 1, 1, 4);
    assert!(
        (diff.to_f64() - (-1.0e-15)).abs() < 1.0e-28,
        "quad mass difference lost precision: got {:.16e}",
        diff.to_f64()
    );

    // Check pass-fraction identity and conservation under accretion & loss.
    for (old_raw, dm_new_raw, delta_m_val) in [
        (
            [0.25_f64, 0.5, 1.5, 1.0],
            [1.0_f64, 1.0, 1.5, 1.0],
            1.25_f64,
        ),
        (
            [2.0_f64, 1.75, 1.5, 1.0],
            [1.0_f64, 1.5, 1.5, 1.0],
            -1.25_f64,
        ),
    ] {
        let nz = old_raw.len();
        let dm_new = one_based(&dm_new_raw);
        let dm_old = one_based(&old_raw);
        let ci = make_compressed_intersect(&dm_new, &dm_old, nz);
        let (i_min, i_max) = find_i_ranges(nz, &ci.ranges);
        assert_eq!(i_min.len(), nz + 1);
        assert_eq!(i_max.len(), nz + 1);

        let change: Vec<f64> = dm_new_raw
            .iter()
            .zip(&old_raw)
            .map(|(n, o)| n - o)
            .collect();
        let flux_f64 = find_mass_flux(&change);
        let mut flux = zero_slot(nz + 1);
        for k in 1..=nz + 1 {
            flux[k] = Quad::from(flux_f64[k - 1]);
        }

        let delta_m = Quad::from(delta_m_val);
        let pf = prepare_pass_fraction(
            nz,
            delta_m,
            &dm_new,
            &ci.mesh_intersects,
            &ci.ranges,
        );
        let (j_min, j_max) = find_j_ranges(nz, &ci.ranges, &flux);

        for i in 1..=nz {
            assert!(j_min[i] <= j_max[i], "j_ranges invalid at cell {i}");
            let inflow = if flux[i] > Quad::ZERO {
                flux[i]
            } else {
                Quad::ZERO
            };
            let outflow = if flux[i + 1] < Quad::ZERO {
                flux[i + 1]
            } else {
                Quad::ZERO
            };
            let expected_through = dm_old[i] + inflow - outflow;
            let mut sum_pf = Quad::ZERO;
            if delta_m_val < 0.0 {
                sum_pf += (-delta_m) * compute_pass_fraction(i as i32, 0, &pf);
            }
            for j in 1..=nz {
                sum_pf += dm_new[j] * compute_pass_fraction(i as i32, j, &pf);
            }
            assert!(
                (sum_pf - expected_through).to_f64().abs() < 1.0e-15,
                "pass_fraction sum mismatch at cell {i}: got {}, want {}",
                sum_pf.to_f64(),
                expected_through.to_f64()
            );
        }

        if delta_m_val > 0.0 {
            let vals_old = vec![
                vec![0.0, 0.0],
                vec![0.70, 0.30],
                vec![0.50, 0.50],
                vec![0.20, 0.80],
                vec![0.10, 0.90],
            ];
            let vals_out = [0.75_f64, 0.25_f64];
            let vals_new = integrate_conserved(
                &vals_old, &vals_out, &dm_new, &dm_old, nz, 2,
            );
            for spec in 0..2 {
                let m_before: f64 =
                    (1..=nz).map(|k| old_raw[k - 1] * vals_old[k][spec]).sum();
                let m_after: f64 = (1..=nz)
                    .map(|j| dm_new_raw[j - 1] * vals_new[j][spec])
                    .sum();
                assert!(
                    rel_err(m_after, m_before + delta_m_val * vals_out[spec])
                        < 1.0e-14
                );
            }
        }
    }
}

#[test]
fn adjust_mass_zero_delta_m_resets_k_below_just_added_and_step_hist() {
    let mut model = zams_model(2.0);
    let old_xmstar = model.xmstar;
    let dt = 10.0 * SECYER;
    model.mstar_dot = 1.0e-4 * M_SUN / SECYER;
    model.dt = dt;
    do_adjust_mass_step(&mut model, dt).expect("accretion step");

    assert!(model.k_below_just_added > 1);
    assert!(model.aux.k_below_const_q > 1);
    assert!(model.aux.k_const_mass > 0);
    assert!(model.eps_mdot.iter().any(|&v| v != 0.0));
    assert_ne!(model.aux.step_hist.total_energy_before_adjust_mass, 0.0);
    assert_ne!(model.aux.step_hist.total_energy_after_adjust_mass, 0.0);
    // `revise_q_and_dq` updates `max_q_for_k_below_const_q = mold_o_mnew`.
    assert!(
        (model.controls.mix_wind.max_q_for_k_below_const_q
            - old_xmstar / model.xmstar)
            .abs()
            < 1.0e-15
    );

    // Follow with zero mass change -> upstream initializes `k_below_just_added = 1`
    // before checking `delta_m == 0` (`adjust_mass.f90:212-230`).
    model.save_old_values();
    model.mstar_dot = 0.0;
    do_adjust_mass_step(&mut model, dt).expect("zero mdot step");

    assert_eq!(model.k_below_just_added, 1);
    assert_eq!(model.aux.k_below_const_q, 1);
    assert_eq!(model.aux.k_const_mass, 0);
    assert!(model.eps_mdot.iter().all(|&v| v == 0.0));
    assert_eq!(model.aux.step_hist.total_energy_before_adjust_mass, 0.0);
    assert_eq!(model.aux.step_hist.total_energy_after_adjust_mass, 0.0);
    assert_eq!(model.aux.step_hist.mdot_acoustic_surface, 0.0);
    assert_eq!(model.aux.step_hist.mdot_adiabatic_surface, 0.0);

    // Same reset invariant on explicit `do_adjust_mass(..., delta_m = 0.0, ...)`
    // and `dt = 0.0` `do_adjust_mass_step`.
    model.mstar_dot = 1.0e-4 * M_SUN / SECYER;
    do_adjust_mass_step(&mut model, dt).expect("second accretion step");
    assert!(model.k_below_just_added > 1);
    do_adjust_mass(&mut model, 0.0, None, 0.99, 0.95).expect("zero delta_m");
    assert_eq!(model.k_below_just_added, 1);
    assert_eq!(model.aux.k_below_const_q, 1);
    assert_eq!(model.aux.k_const_mass, 0);

    model.mstar_dot = 1.0e-4 * M_SUN / SECYER;
    do_adjust_mass_step(&mut model, dt).expect("third accretion step");
    assert!(model.k_below_just_added > 1);
    do_adjust_mass_step(&mut model, 0.0).expect("dt = 0 step");
    assert_eq!(model.k_below_just_added, 1);
}

#[test]
fn adjust_mass_conserves_accreted_species_interior_xa_and_cell_volumes() {
    // 1. Accretion composition conservation with explicit `x_acc`.
    let mut acc = zams_model(2.0);
    let nz = acc.nz;
    let h1_idx = acc.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = acc.iso_names.iter().position(|n| n == "he4").expect("he4");
    let dm_old = acc.dm[..nz].to_vec();
    let xa_old = acc.xa.clone();
    let h1_before: f64 =
        (0..nz).map(|k| dm_old[k] * acc.cell_xa(k)[h1_idx]).sum();
    let he4_before: f64 =
        (0..nz).map(|k| dm_old[k] * acc.cell_xa(k)[he4_idx]).sum();

    let delta_m = 5.0e-4 * M_SUN;
    acc.controls.accrete_same_as_surface = false;
    acc.controls.accrete_given_mass_fractions = true;
    acc.controls.mix_wind.num_accretion_species = 2;
    acc.controls.accretion_species_id =
        vec![String::from("h1"), String::from("he4")];
    acc.controls.accretion_species_xa = vec![0.20, 0.80];
    acc.mstar_dot = delta_m / SECYER;

    let mut acc_direct = acc.clone();
    do_adjust_mass_step(&mut acc, SECYER).expect("accretion step");

    let h1_after: f64 =
        (0..nz).map(|k| acc.dm[k] * acc.cell_xa(k)[h1_idx]).sum();
    let he4_after: f64 =
        (0..nz).map(|k| acc.dm[k] * acc.cell_xa(k)[he4_idx]).sum();
    let err_h1 = rel_err(h1_after - h1_before, 0.20 * delta_m);
    assert!(
        err_h1 < 1.0e-9,
        "h1 mass mismatch: got {:.12e}, want {:.12e} (rel={err_h1:.3e})",
        h1_after - h1_before,
        0.20 * delta_m
    );
    assert!(rel_err(h1_after, h1_before + 0.20 * delta_m) < 1.0e-14);
    let err_he4 = rel_err(he4_after - he4_before, 0.80 * delta_m);
    assert!(
        err_he4 < 1.0e-9,
        "he4 mass mismatch: got {:.12e}, want {:.12e} (rel={err_he4:.3e})",
        he4_after - he4_before,
        0.80 * delta_m
    );
    assert!(rel_err(he4_after, he4_before + 0.80 * delta_m) < 1.0e-14);
    assert!((acc.cell_xa(0)[he4_idx] - 0.80).abs() < 1.0e-12);

    // Region III interior cells (`k > k_const_mass`) remain unmodified.
    let species = acc.species;
    for k in (acc.aux.k_const_mass + 1)..nz {
        assert_eq!(
            &acc.xa[k * species..(k + 1) * species],
            &xa_old[k * species..(k + 1) * species],
            "interior zone {k} xa changed past k_const_mass"
        );
    }

    // 2. Explicit `do_adjust_mass` keeps `update_radius` cumulative volumes exact
    //    (`(4pi/3)*r(j)^3 == vol_center + sum_{l>=j} dm(l)/rho(l)`) and `ln_r_start == ln_r`.
    do_adjust_mass(&mut acc_direct, delta_m, None, 0.99, 0.95)
        .expect("explicit do_adjust_mass");
    let rc = acc_direct.r_center;
    let mut vol_cum = FOUR_THIRDS_PI * rc * rc * rc;
    for j in (0..nz).rev() {
        vol_cum += acc_direct.dm[j] / acc_direct.rho[j];
        let r0 = flexmesa_math::exp(acc_direct.ln_r[j]);
        let r_vol = FOUR_THIRDS_PI * r0 * r0 * r0;
        assert!(
            rel_err(r_vol, vol_cum) < 1.0e-13,
            "update_radius volume mismatch at cell {j}: got {r_vol:.6e}, want {vol_cum:.6e}"
        );
        assert_eq!(
            acc_direct.ln_r_start[j].to_bits(),
            acc_direct.ln_r[j].to_bits()
        );
    }
    assert_eq!(
        acc_direct.q[nz - 1].to_bits(),
        acc_direct.dq[nz - 1].to_bits()
    );
}

#[test]
fn adjust_mass_rotation_set1_omega_and_adjust_j_lost_parity() {
    // 1. `use_accreted_material_j = true` exact total angular momentum conservation.
    let mut m_acc = zams_model(2.0);
    let nz = m_acc.nz;
    m_acc.controls.rotation_flag = true;
    m_acc.controls.mix_wind.do_adjust_j_lost = false;
    m_acc.controls.mix_wind.use_accreted_material_j = true;
    m_acc.controls.mix_wind.accreted_material_j = 5.0e16;
    for k in 0..nz {
        m_acc.omega[k] = 1.0e-5 * (1.0 + 0.25 * (k as f64 / nz as f64));
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m_acc);
    m_acc.save_old_values();

    let j_before = total_angular_momentum(&m_acc);
    let dm_gain = 1.0e-3 * M_SUN;
    m_acc.mstar_dot = dm_gain / SECYER;
    do_adjust_mass_step(&mut m_acc, SECYER).expect("accrete with j");
    let j_after = total_angular_momentum(&m_acc);
    assert!(
        rel_err(j_after - j_before, dm_gain * 5.0e16) < 1.0e-11,
        "total angular momentum delta mismatch: got {:.12e}, want {:.12e}",
        j_after - j_before,
        dm_gain * 5.0e16
    );

    // 2. `adjust_j_lost` extending inward past `k_const_mass` must integrate
    //    with updated `model.dm` (not stale `new_cell_mass`) and match
    //    `delta_j = actual_j_lost - j_removed` exactly.
    let mut base = zams_model(2.0);
    base.controls.rotation_flag = true;
    base.controls.mix_wind.adjust_j_fraction = 1.0;
    base.controls.mix_wind.min_q_for_adjust_j_lost = 0.92;
    base.controls.mix_wind.min_j_div_delta_j = 10.0;
    for k in 0..base.nz {
        base.omega[k] = 1.0e-5 * (1.0 + 0.5 * (k as f64 / base.nz as f64));
    }
    use_xh_to_update_i_rot_and_j_rot(&mut base);
    base.save_old_values();

    let dm_loss = -0.01 * M_SUN;
    let starting_j_surf = base.j_rot[0];
    let old_dm_bar = dm_bar_from_dm(&base.dm);
    let mut j_removed = 0.0_f64;
    let mut dm_acc = 0.0_f64;
    let lost = -dm_loss;
    for k in 0..base.nz {
        let mut take = old_dm_bar[k];
        if dm_acc + take > lost {
            take = lost - dm_acc;
            dm_acc = lost;
        } else {
            dm_acc += take;
        }
        j_removed += take * base.j_rot[k];
        if dm_acc == lost {
            break;
        }
    }
    let delta_j_expected = lost * starting_j_surf - j_removed;

    let mut no_adj = base.clone();
    no_adj.controls.mix_wind.do_adjust_j_lost = false;
    do_adjust_mass(&mut no_adj, dm_loss, None, 0.998, 0.995)
        .expect("no adjust_j_lost");

    let mut with_adj = base.clone();
    with_adj.controls.mix_wind.do_adjust_j_lost = true;
    do_adjust_mass(&mut with_adj, dm_loss, None, 0.998, 0.995)
        .expect("with adjust_j_lost");

    assert!(
        with_adj.adjust_j_q < 0.995,
        "expected last_k inside k_const_mass, adjust_j_q={}",
        with_adj.adjust_j_q
    );
    let post_dm_bar = dm_bar_from_dm(&with_adj.dm);
    let delta_j_applied: f64 = (0..with_adj.nz)
        .map(|k| post_dm_bar[k] * (no_adj.j_rot[k] - with_adj.j_rot[k]))
        .sum();
    assert!(
        rel_err(delta_j_applied, delta_j_expected) < 1.0e-12,
        "adjust_j_lost DM mismatch across k_const_mass: got {delta_j_applied:.12e}, want {delta_j_expected:.12e}"
    );

    // 3. Early returns in `adjust_j_lost` (`delta_j == 0` and `mass_lost <= 0`)
    //    must reset `adjust_j_q` to `1.0` (`adjust_mass.f90:1493, 1515`).
    let mut m_zero_dj = with_adj.clone();
    m_zero_dj.controls.mix_wind.adjust_j_fraction = 0.0;
    do_adjust_mass(&mut m_zero_dj, -1.0e-5 * M_SUN, None, 0.998, 0.995)
        .expect("zero delta_j step");
    assert_eq!(m_zero_dj.adjust_j_q, 1.0);

    let mut m_gain = with_adj;
    do_adjust_mass(&mut m_gain, 1.0e-5 * M_SUN, None, 0.998, 0.995)
        .expect("mass gain resets adjust_j_q");
    assert_eq!(m_gain.adjust_j_q, 1.0);
}

#[test]
fn accretion_xa_scales_custom_missing_metals_and_errors_parity() {
    let mut model = zams_model(2.0);
    let catalog = NuclideCatalog::builtin_ref();
    let h1_idx = model.iso_names.iter().position(|n| n == "h1").expect("h1");
    let he4_idx = model
        .iso_names
        .iter()
        .position(|n| n == "he4")
        .expect("he4");
    let c12_idx = model
        .iso_names
        .iter()
        .position(|n| n == "c12")
        .expect("c12");
    let o16_idx = model
        .iso_names
        .iter()
        .position(|n| n == "o16")
        .expect("o16");

    model.controls.accrete_same_as_surface = false;
    model.controls.accrete_given_mass_fractions = false;
    model.controls.accretion_h1 = 0.70;
    model.controls.mix_wind.accretion_h2 = 1.0e-5;
    model.controls.mix_wind.accretion_he3 = 3.0e-5;
    model.controls.accretion_he4 = 0.279_96;
    model
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = false;

    for id in 1_usize..=10 {
        model.controls.mix_wind.accretion_zfracs =
            i32::try_from(id).expect("1..=10");
        let xa = get_xa_for_accretion(&model).expect("solar zfrac scale");
        assert!((xa.iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        // `h2` is absent in `basic.net`, so `h1 + h2` maps to `xa[h1]`.
        assert!((xa[h1_idx] - (0.70 + 1.0e-5)).abs() < 1.0e-13);
        assert!((xa[he4_idx] - 0.279_96).abs() < 1.0e-13);

        let scale = SolarScale::from_id(id).expect("scale");
        let zfrac = solar_element_zfrac(scale, catalog);
        let c_iso_pct = lodders03_element_atom_percent("c12")
            / (lodders03_element_atom_percent("c12")
                + model
                    .iso_names
                    .iter()
                    .position(|n| n == "c13")
                    .map_or(0.0, |_| {
                        lodders03_element_atom_percent("c13") * 13.0 / 12.0
                    }));
        assert!(zfrac[5] > 0.0 && c_iso_pct > 0.0 && xa[o16_idx] > 0.0);
    }

    // Custom metal mixture with `dump_missing_metals_into_heaviest` true vs false.
    model.controls.mix_wind.accretion_zfracs = 0;
    model.controls.mix_wind.accretion_h2 = 0.0;
    model.controls.mix_wind.accretion_he3 = 0.0;
    model.controls.accretion_he4 = 0.28;
    model.controls.mix_wind.z_fraction_c = 0.20;
    model.controls.mix_wind.z_fraction_o = 0.50;
    model.controls.mix_wind.z_fraction_fe = 0.30; // `fe` missing in basic net!

    model
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = false;
    let xa_renorm = get_xa_for_accretion(&model).expect("custom renorm");
    assert!(
        rel_err(xa_renorm[c12_idx] / xa_renorm[o16_idx], 0.20 / 0.50) < 1e-12
    );
    assert!((xa_renorm[c12_idx] + xa_renorm[o16_idx] - 0.02).abs() < 1e-12);

    model
        .controls
        .mix_wind
        .accretion_dump_missing_metals_into_heaviest = true;
    let xa_heavy = get_xa_for_accretion(&model).expect("custom dump heaviest");
    let heaviest_idx = model
        .chem_id
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let wa = catalog.get_by_id(**a).map_or(0.0, |i| i.w);
            let wb = catalog.get_by_id(**b).map_or(0.0, |i| i.w);
            wa.total_cmp(&wb)
        })
        .expect("heaviest")
        .0;
    assert!((xa_heavy[c12_idx] - 0.02 * 0.20).abs() < 1.0e-12);
    assert!((xa_heavy[o16_idx] - 0.02 * 0.50).abs() < 1.0e-12);
    assert!((xa_heavy[heaviest_idx] - 0.02 * 0.30).abs() < 1.0e-12);

    // Invalid scale is accepted iff metals == 0, rejected when metals > 0.
    model.controls.mix_wind.accretion_zfracs = 999;
    assert!(matches!(
        get_xa_for_accretion(&model),
        Err(StarError::InvalidInput(_))
    ));
    let mut zero_z = model.clone();
    zero_z.controls.accretion_h1 = 0.72;
    zero_z.controls.accretion_he4 = 0.28;
    let xa_z0 = get_xa_for_accretion(&zero_z).expect("zero metals allowed");
    assert!((xa_z0[h1_idx] - 0.72).abs() < 1.0e-14);
    assert!((xa_z0[he4_idx] - 0.28).abs() < 1.0e-14);

    // Infallible legacy wrapper falls back to surface composition on error.
    let fallback = build_accreted_xa(&model);
    assert_eq!(fallback, model.cell_xa(0));

    // Negative `z_fraction_*` returns `InvalidInput`.
    model.controls.mix_wind.accretion_zfracs = 0;
    model.controls.mix_wind.z_fraction_c = -0.1;
    model.controls.mix_wind.z_fraction_fe = 0.6;
    assert!(matches!(
        get_xa_for_accretion(&model),
        Err(StarError::InvalidInput(_))
    ));
}

#[test]
fn eps_mdot_early_return_clears_step_hist_and_sets_need_to_setvars() {
    let mut model = zams_model(2.0);
    let nz = model.nz;
    let dm_before = model.dm[..nz].to_vec();
    let e_before =
        total_energy_profile_before_adjust_mass(&model).expect("e_before");
    let record = AdjustMassRecord {
        dm_before,
        total_energy_profile_before: e_before,
        outer_frac_sub1: 0.0,
        mid_frac_sub1: 0.0,
        inner_frac_sub1: 0.0,
        k_below_const_q: 5,
        k_const_mass: 10,
    };

    // Dirty `eps_mdot` and the 8 `step_hist` scalars cleared by `eps_mdot.f90:494-505`.
    model.eps_mdot.fill(42.0);
    {
        let sh = &mut model.aux.step_hist;
        sh.mdot_adiabatic_surface = 1.0;
        sh.mdot_acoustic_surface = 2.0;
        sh.total_internal_energy_after_adjust_mass = 3.0;
        sh.total_gravitational_energy_after_adjust_mass = 4.0;
        sh.total_radial_kinetic_energy_after_adjust_mass = 5.0;
        sh.total_rotational_kinetic_energy_after_adjust_mass = 6.0;
        sh.total_turbulent_energy_after_adjust_mass = 7.0;
        sh.total_energy_after_adjust_mass = 8.0;
    }
    model.aux.need_to_setvars = false;
    model.mstar_dot = -1.0e-5 * M_SUN / SECYER;

    // `dt <= 0.0` early return clears scalars and leaves `need_to_setvars = false`.
    let surf0 = calculate_eps_mdot(&mut model, &record, 0.0).expect("dt=0");
    assert_eq!(surf0.mdot_acoustic_surface, 0.0);
    assert_eq!(surf0.mdot_adiabatic_surface, 0.0);
    assert!(model.eps_mdot.iter().all(|&v| v == 0.0));
    assert!(!model.aux.need_to_setvars);
    let sh = &model.aux.step_hist;
    assert_eq!(sh.mdot_adiabatic_surface, 0.0);
    assert_eq!(sh.mdot_acoustic_surface, 0.0);
    assert_eq!(sh.total_internal_energy_after_adjust_mass, 0.0);
    assert_eq!(sh.total_gravitational_energy_after_adjust_mass, 0.0);
    assert_eq!(sh.total_radial_kinetic_energy_after_adjust_mass, 0.0);
    assert_eq!(sh.total_rotational_kinetic_energy_after_adjust_mass, 0.0);
    assert_eq!(sh.total_turbulent_energy_after_adjust_mass, 0.0);
    assert_eq!(sh.total_energy_after_adjust_mass, 0.0);

    // Active mass loss sets `need_to_setvars = true` (`eps_mdot.f90:507`) and scales
    // with `eps_mdot_factor` (`eps_mdot.f90:666`).
    let mut step_m = zams_model(2.0);
    step_m.mstar_dot = -1.0e-4 * M_SUN / SECYER;
    do_adjust_mass_step(&mut step_m, SECYER).expect("active eps_mdot");
    assert!(step_m.aux.need_to_setvars);
    let nonzero_count = step_m.eps_mdot.iter().filter(|&&v| v != 0.0).count();
    assert!(nonzero_count > 0);
    assert_eq!(
        eval_total_energy_profile(&step_m)
            .expect("eval_total_energy_profile")
            .len(),
        nz
    );

    let mut step_fac0 = zams_model(2.0);
    step_fac0.mstar_dot = -1.0e-4 * M_SUN / SECYER;
    step_fac0.controls.mix_wind.eps_mdot_factor = 0.0;
    do_adjust_mass_step(&mut step_fac0, SECYER).expect("factor 0");
    assert!(step_fac0.aux.need_to_setvars);
    assert!(step_fac0.eps_mdot.iter().all(|&v| v == 0.0));
}

#[test]
fn mdot_redo_surf_avg_roche_negative_lrad_tau_edge_and_state_machine() {
    // 1. Fast rotation (`w_div_wc ~ 0.78`), inverted surface temperature (negative
    //    `Lrad/Ledd` -> `gamma_factor > 1`), and optical depth edge cases.
    let mut rot = zams_model(2.0);
    rot.controls.rotation_flag = true;
    for k in 0..rot.nz {
        rot.omega[k] = 0.78 * eval_upstream_omega_crit(&rot, k);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut rot);

    let w_redo = eval_surf_avg_w_div_w_crit(&mut rot);
    let info = eval_surf_avg_rotation_info(&rot);
    assert_eq!(w_redo.to_bits(), info.w_div_w_crit.to_bits());

    // Independent reference calculation using Roche equatorial radii & signed Lrad/Ledd.
    let tau0 = rot.controls.tau_factor * rot.tau_base;
    let tau_min = rot.controls.output.surf_avg_tau_min;
    let tau_max = rot.controls.output.surf_avg_tau;
    rot.t[0] = rot.t[1] * 1.25; // negative `get_lrad_div_ledd(rot, 0)`
    rot.controls.tau_factor = 0.0;
    rot.controls.output.surf_avg_tau_min = 0.0;
    rot.controls.output.surf_avg_tau = 1.0e-6;
    rot.controls.mix_wind.surf_avg_tau_min = 0.0;
    rot.controls.mix_wind.surf_avg_tau = 1.0e-6;
    set_rotation_info(&mut rot, true);
    let lrad_neg = get_lrad_div_ledd(&rot, 0);
    assert!(
        lrad_neg < 0.0,
        "expected negative face Lrad/Ledd, got {lrad_neg}"
    );
    let w_neg_lrad = eval_surf_avg_w_div_w_crit(&mut rot);
    let info_neg = eval_surf_avg_rotation_info(&rot);
    assert_eq!(w_neg_lrad.to_bits(), info_neg.w_div_w_crit.to_bits());
    assert!(info_neg.lrad_div_ledd < 0.0);

    // Restore optical depth bounds and check `mix_wind.surf_avg_tau` override
    // forwarding when `output` is default, plus `tau0 >= surf_avg_tau`.
    rot.controls.tau_factor = 1.0;
    rot.controls.output.surf_avg_tau_min = tau_min;
    rot.controls.output.surf_avg_tau = tau_max;
    rot.controls.mix_wind.surf_avg_tau_min = tau_min;
    rot.controls.mix_wind.surf_avg_tau = 50.0;
    let w_mw_override = eval_surf_avg_w_div_w_crit(&mut rot);
    assert!(w_mw_override.is_finite() && w_mw_override > 0.0);
    assert_eq!(rot.controls.output.surf_avg_tau, 100.0);

    rot.controls.mix_wind.surf_avg_tau = 100.0;
    rot.controls.tau_factor = 300.0; // tau0 = 200 > 100
    assert!(tau0 < tau_max && rot.controls.tau_factor * rot.tau_base > tau_max);
    let w_surf_only = eval_surf_avg_w_div_w_crit(&mut rot);
    let gamma_0 = 1.0 - get_lrad_div_ledd(&rot, 0).min(0.9999);
    let rmid_0 = 0.5
        * (equatorial_and_polar_r_cm(&rot, 0).0
            + equatorial_and_polar_r_cm(&rot, 1).0);
    let cgrav_0 = 0.5 * (rot.cgrav[0] + rot.cgrav[1]);
    let mmid_0 = 0.5 * (rot.m_grav[0] + rot.m_grav[1]);
    let om_0 = 0.5 * (rot.omega[0] + rot.omega[1]);
    let om_c0 =
        (gamma_0 * cgrav_0 * mmid_0 / (rmid_0 * rmid_0 * rmid_0)).sqrt();
    let w_expected_0 = (om_0 / om_c0).abs();
    assert!(rel_err(w_surf_only, w_expected_0) < 1.0e-14);
    let _ = tau_min;

    // 2. `MdotRedoState::setup` `ignored_first_step` trigger does not gate on
    //    `max_mdot_redo_cnt > 0`, and `select_action` returns `Accept` cleanly
    //    when `max_mdot_redo_cnt == 0` (`evolve.f90:766-775, 794`).
    let mut m_ign = zams_model(2.0);
    m_ign.generations = 2;
    m_ign.controls.rotation_flag = true;
    m_ign.controls.mix_wind.max_mdot_redo_cnt = 0;
    m_ign.controls.mix_wind.mdot_revise_factor = 1.1;
    m_ign.aux.step_hist.was_in_implicit_wind_limit = true;
    let mdot_old = -1.0e-5 * M_SUN / SECYER;
    m_ign.aux.step_hist.mstar_dot_old = mdot_old;
    m_ign.mstar_dot = -1.0e-7 * M_SUN / SECYER;
    let mut redo_ign = MdotRedoState::setup(&mut m_ign);
    assert_eq!(m_ign.mstar_dot, mdot_old);
    assert_eq!(
        redo_ign.select_action(&mut m_ign, SECYER),
        SelectMdotAction::Accept
    );

    // 3. `iwind` implicit wind iteration with `StarExtras::other_wind`, plus
    //    `rotational_mdot_boost_old` preservation.
    let mut m_iw = zams_model(2.0);
    m_iw.controls.rotation_flag = false;
    m_iw.controls.mix_wind.max_mdot_redo_cnt = 3;
    m_iw.controls.max_tries_for_implicit_wind = 3;
    m_iw.controls.iwind_tolerance = 1.0e-3;
    m_iw.controls.iwind_lambda = 0.5;
    m_iw.controls.use_other_wind = true;
    m_iw.controls.mix_wind.hot_wind_scheme = String::from("other");
    m_iw.controls.cool_wind_rgb_scheme = String::from("other");
    m_iw.controls.cool_wind_agb_scheme = String::from("other");
    m_iw.aux.step_hist.rotational_mdot_boost_old = 1.23;
    let start_mdot = -1.0e-7 * M_SUN / SECYER;
    let target_wind = 3.0e-7_f64; // Msun/yr -> mstar_dot = -3e-7 Msun/yr
    let target_mdot = -target_wind * M_SUN / SECYER;
    m_iw.mstar_dot = start_mdot;
    let mut st_iw = MdotRedoState::setup(&mut m_iw);
    let act = st_iw
        .select_action_with_extras(
            &mut m_iw,
            SECYER,
            &mut FixedWind(target_wind),
        )
        .expect("iwind redo");
    let expected_next = start_mdot + 0.5 * (target_mdot - start_mdot);
    assert_eq!(act, SelectMdotAction::RedoWithMdot(expected_next));
    assert_eq!(m_iw.aux.step_hist.rotational_mdot_boost_old, 1.23);

    // 4. `w > surf_limit` -> bracket -> bisect -> `Accept`, and `compute_delta_m`
    //    zeroing at mass limit.
    let mut m_bi = zams_model(2.0);
    m_bi.controls.rotation_flag = true;
    m_bi.controls.surf_omega_div_omega_crit_limit = 0.70;
    m_bi.controls.mix_wind.surf_omega_div_omega_crit_tol = 0.05;
    m_bi.controls.mix_wind.max_mdot_redo_cnt = 10;
    m_bi.controls.mix_wind.mdot_revise_factor = 2.0;
    m_bi.controls.mix_wind.rotational_mdot_boost_fac = 1.0e10;
    m_bi.controls.mix_wind.rotational_mdot_kh_fac = 1.0e10;
    for k in 0..m_bi.nz {
        m_bi.omega[k] = 0.80 * eval_upstream_omega_crit(&m_bi, k);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m_bi);
    m_bi.save_old_values();
    let mdot0 = -1.0e-6 * M_SUN / SECYER;
    m_bi.mstar_dot = mdot0;
    let mut st_bi = MdotRedoState::setup(&mut m_bi);
    let act1 = st_bi.select_action(&mut m_bi, SECYER);
    assert_eq!(act1, SelectMdotAction::RedoWithMdot(2.0 * mdot0));
    assert!(m_bi.aux.step_hist.was_in_implicit_wind_limit);

    // Next trial drops rotation below limit -> brackets & bisects.
    m_bi.mstar_dot = 2.0 * mdot0;
    for k in 0..m_bi.nz {
        m_bi.omega[k] = 0.55 * eval_upstream_omega_crit(&m_bi, k);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m_bi);
    let act2 = st_bi.select_action(&mut m_bi, SECYER);
    assert_eq!(act2, SelectMdotAction::RedoWithMdot(1.5 * mdot0));

    // Bisected midpoint still below limit -> `(2.0 - 1.0)/2.0 = 0.5 >= tol`,
    // tightens `too_large` to `1.5`, bisects to `1.25`.
    m_bi.mstar_dot = 1.5 * mdot0;
    let act3 = st_bi.select_action(&mut m_bi, SECYER);
    assert_eq!(act3, SelectMdotAction::RedoWithMdot(1.25 * mdot0));

    // At max retries -> `ForceRetry`.
    m_bi.controls.mix_wind.max_mdot_redo_cnt = 1;
    let mut m_max = zams_model(2.0);
    m_max.controls.rotation_flag = true;
    m_max.controls.surf_omega_div_omega_crit_limit = 0.50;
    m_max.controls.mix_wind.max_mdot_redo_cnt = 1;
    m_max.controls.mix_wind.rotational_mdot_boost_fac = 1.0e10;
    m_max.controls.mix_wind.rotational_mdot_kh_fac = 1.0e10;
    for k in 0..m_max.nz {
        m_max.omega[k] = 0.80 * eval_upstream_omega_crit(&m_max, k);
    }
    use_xh_to_update_i_rot_and_j_rot(&mut m_max);
    m_max.mstar_dot = mdot0;
    let mut st_max = MdotRedoState::setup(&mut m_max);
    assert_eq!(
        st_max.select_action(&mut m_max, SECYER),
        SelectMdotAction::RedoWithMdot(1.1 * mdot0)
    );
    m_max.mstar_dot = 1.1 * mdot0;
    assert_eq!(
        st_max.select_action(&mut m_max, SECYER),
        SelectMdotAction::ForceRetry
    );

    // Sanity check `compute_delta_m` stops at `max_star_mass_for_gain`.
    let mut m_lim = zams_model(2.0);
    m_lim.controls.mix_wind.max_star_mass_for_gain = 1.9;
    m_lim.mstar_dot = 1.0e-5 * M_SUN / SECYER;
    assert_eq!(compute_delta_m(&mut m_lim, SECYER).expect("dm"), 0.0);
}
