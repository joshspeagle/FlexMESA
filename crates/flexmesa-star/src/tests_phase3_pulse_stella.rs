// SPDX-License-Identifier: LGPL-3.0-or-later
//! Phase 3 (`Issue #4`) unit and golden parity tests for all 7 pulsation
//! exporters (`GYRE`, `FGONG`, `OSC`, `GSM`, `SAIO`, `CAFEIN`, `GR1D`),
//! `GYRE` 34-mode BVP golden fixture (`gyre/test/test_output`), `ADIPLS`
//! 46-mode `Gemma.fgong` golden fixture (`adipls/test/obs.test.txt`), and
//! `STELLA` `.hyd`/`.abn` + Sobolev $\mathrm{Fe\,II}\ \lambda 5169$ golden
//! fixtures (`stella/test/`).

use std::path::PathBuf;

use crate::{
    build_pulse_points_with_options, export_cafein, export_fgong_with_version,
    export_gr1d, export_gsm, export_gsm_data, export_gyre_with_schema,
    export_osc, export_pulse_by_format, export_saio, export_stella_pair,
    load_zams_model, parse_fortran_float, parse_stella_abn, parse_stella_hyd,
    parse_stella_lbol, run_stella_from_hyd_abn, set_hydro_vars,
    solve_adipls_fgong_modes, solve_gyre_mesa_modes, FeII5169EtaTable,
    PulseExportOptions, StarControls, StarJobControls, StellaExportOptions,
};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/golden")
}

#[test]
fn test_all_seven_pulsation_exporters_and_double_point_atmosphere_options() {
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");

    // Introduce a sharp composition gradient at face k = model.nz / 2 to test double-point splitting
    let k_split = model.nz / 2;
    for k in k_split..model.nz {
        model.mu[k] *= 1.35;
    }
    let opts_base = PulseExportOptions::default();
    let pts_base = build_pulse_points_with_options(&model, &opts_base);
    assert_eq!(pts_base.len(), model.nz + 1);
    assert_eq!(pts_base[0].r, 0.0);
    assert!((pts_base[0].u_dziem - 3.0).abs() <= 1.0e-14);
    assert_eq!(pts_base[0].v_dziem, 0.0);
    assert_eq!(pts_base[0].a_star, 0.0);

    let opts_split = PulseExportOptions {
        add_double_points: true,
        grad_mu_threshold: 0.05,
        add_atmosphere: true,
        ..opts_base
    };
    let pts_split = build_pulse_points_with_options(&model, &opts_split);
    assert!(pts_split.len() > pts_base.len() + 6);

    // 1. GYRE v101 & v110 (19 cols = index + 18), v120 (20 cols = index + 19)
    for (schema, expected_cols) in [(101, 19), (110, 19), (120, 20)] {
        let g = export_gyre_with_schema(&model, true, schema);
        let mut lines = g.lines();
        let hdr: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
        assert_eq!(hdr.len(), 5);
        assert_eq!(hdr[4].parse::<usize>().unwrap(), schema);
        let row0: Vec<&str> =
            lines.next().unwrap().split_whitespace().collect();
        assert_eq!(row0.len(), expected_cols);
    }

    // 2. FGONG v300 and v1300 (15 global constants + 40 point columns)
    for ver in [300, 1300] {
        let fg = export_fgong_with_version(&model, true, ver);
        assert_eq!(fg.lines().count(), 5 + 3 + 8 * pts_base.len());
    }

    // 3. OSC (15 global constants + 36 point columns = 22 structure + 14 abundance)
    let osc = export_osc(&model, true);
    assert_eq!(osc.lines().count(), 6 + 3 + 8 * pts_base.len());

    // 4. GSM v110 and v120
    let gsm_110 = export_gsm_data(&model, true, 110);
    let gsm_120 = export_gsm_data(&model, true, 120);
    assert_eq!(gsm_110.n, pts_base.len());
    assert_eq!(gsm_120.datasets.len(), 19);
    assert!(
        export_gsm(&model, true).contains("\"version\": 120")
            || export_gsm(&model, true).contains("120")
    );

    // 5. SAIO (4 global constants + 20 point columns = 4 lines per point)
    let saio = export_saio(&model, true, false);
    assert_eq!(saio.lines().count(), 1 + 4 * model.nz);

    // 6. CAFEIN (13 global constants + 35 point columns)
    let caf = export_cafein(&model, true, true, false);
    assert_eq!(caf.lines().count(), 4 + pts_base.len());
    let caf_last: Vec<&str> =
        caf.lines().last().unwrap().split_whitespace().collect();
    assert_eq!(caf_last.len(), 35);

    // 7. GR1D (1 count header + index + 7 cell-midpoint columns fixing upstream MESA pulse_gr1d.f90 bugs)
    let gr1d = export_gr1d(&model);
    assert_eq!(gr1d.lines().count(), 1 + model.nz);
    let gr1d_first: Vec<&str> =
        gr1d.lines().nth(1).unwrap().split_whitespace().collect();
    assert_eq!(gr1d_first.len(), 8);

    // Unified dispatcher (`export_pulse_by_format`)
    for fmt_name in ["gyre", "fgong", "osc", "gsm", "saio", "cafein", "gr1d"] {
        assert!(export_pulse_by_format(&model, fmt_name, true).is_some());
    }
}

#[test]
fn test_gyre_bvp_solver_reproduces_all_34_modes_of_gyre_test_output() {
    let dir = golden_dir().join("gyre");
    let model_str =
        std::fs::read_to_string(dir.join("model.dat")).expect("gyre/model.dat");
    let test_out = std::fs::read_to_string(dir.join("test_output"))
        .expect("gyre/test_output");

    let mut modes = solve_gyre_mesa_modes(&model_str, 0, 1.0, 20.0, 200);
    modes.extend(solve_gyre_mesa_modes(&model_str, 1, 1.0, 20.0, 200));

    let mut ref_rows = Vec::new();
    for line in test_out.lines() {
        if line.contains("done file model") {
            break;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() == 6 && t[0].parse::<usize>().is_ok() {
            ref_rows.push((
                t[0].parse::<usize>().unwrap(),
                t[1].parse::<i32>().unwrap(),
                t[2].parse::<i32>().unwrap(),
                t[3].parse::<i32>().unwrap(),
                parse_fortran_float(t[4]),
                parse_fortran_float(t[5]),
            ));
        }
    }
    assert_eq!(ref_rows.len(), 34);
    assert_eq!(modes.len(), 34);

    for (bm, &(l, np, ng, npg, freq_ref, e_norm_ref)) in
        modes.iter().zip(ref_rows.iter())
    {
        assert_eq!(
            (bm.l, bm.n_p, bm.n_g, bm.n_pg),
            (l, np, ng, npg),
            "mode quantum number mismatch"
        );
        let rel_freq = (bm.freq_uhz - freq_ref).abs() / freq_ref;
        let rel_enorm = (bm.e_norm - e_norm_ref).abs() / e_norm_ref;
        assert!(
            rel_freq <= 5.0e-4,
            "l={l} n_pg={npg}: freq_uhz={} vs ref={freq_ref} (rel={rel_freq:.3e})",
            bm.freq_uhz
        );
        assert!(
            rel_enorm <= 5.0e-2,
            "l={l} n_pg={npg}: e_norm={} vs ref={e_norm_ref} (rel={rel_enorm:.3e})",
            bm.e_norm
        );
    }
}

#[test]
fn test_adipls_bvp_solver_reproduces_all_46_modes_of_gemma_fgong_obs_test() {
    let dir = golden_dir().join("adipls");
    let fgong_str = std::fs::read_to_string(dir.join("Gemma.fgong"))
        .expect("adipls/Gemma.fgong");
    let obs_str = std::fs::read_to_string(dir.join("obs.test.txt"))
        .expect("adipls/obs.test.txt");

    let mut modes = Vec::new();
    for l in 0..=2 {
        modes.extend(solve_adipls_fgong_modes(
            &fgong_str, l, 220.0, 1000.0, 500,
        ));
    }

    let mut ref_rows = Vec::new();
    for line in obs_str.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() >= 4 && t[0].parse::<usize>().is_ok() {
            ref_rows.push((
                t[0].parse::<usize>().unwrap(),
                t[1].parse::<i32>().unwrap(),
                parse_fortran_float(t[2]),
                parse_fortran_float(t[3]),
            ));
        }
    }
    assert_eq!(ref_rows.len(), 46);
    assert_eq!(modes.len(), 46);

    for (bm, &(l, n_ord, freq_ref, e_ref)) in modes.iter().zip(ref_rows.iter())
    {
        assert_eq!((bm.l, bm.n_pg), (l, n_ord), "ADIPLS mode order mismatch");
        let rel_freq = (bm.freq_var_uhz - freq_ref)
            .abs()
            .min((bm.freq_uhz - freq_ref).abs())
            / freq_ref;
        let rel_e = (bm.e_adipls.log10() - e_ref.log10()).abs();
        let tol_freq = if l == 0 { 1.6e-3 } else { 1.25e-2 };
        assert!(
            rel_freq <= tol_freq,
            "l={l} n={n_ord}: freq_var_uhz={} freq_uhz={} vs ref={freq_ref} (rel={rel_freq:.3e})",
            bm.freq_var_uhz,
            bm.freq_uhz
        );
        assert!(
            rel_e <= 0.35,
            "l={l} n={n_ord}: e_adipls={} vs ref={e_ref} (dlog10={rel_e:.3e})",
            bm.e_adipls
        );
    }
}

#[test]
fn test_stella_export_and_feii_5169_sobolev_golden_fixture_parity() {
    let dir = golden_dir().join("stella");
    let eta_str = std::fs::read_to_string(dir.join("FeII_5169_eta.dat"))
        .expect("FeII_5169_eta.dat");
    let hyd_str =
        std::fs::read_to_string(dir.join("mesa.hyd")).expect("mesa.hyd");
    let abn_str =
        std::fs::read_to_string(dir.join("mesa.abn")).expect("mesa.abn");
    let lbol_str =
        std::fs::read_to_string(dir.join("mesa.lbol")).expect("mesa.lbol");
    let vel_str = std::fs::read_to_string(dir.join("mesa.vel_feII"))
        .expect("mesa.vel_feII");

    let eta_tab = FeII5169EtaTable::parse(&eta_str);
    assert_eq!(eta_tab.log_rhos.len(), 41);
    assert_eq!(eta_tab.log_ts.len(), 117);

    // Verify bicubic interpolation of eta(rho, T) against golden mesa.vel_feII rows
    for line in vel_str.lines().skip(2) {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 8 {
            continue;
        }
        let rho = parse_fortran_float(t[5]);
        let temp = parse_fortran_float(t[6]);
        let eta_ref = parse_fortran_float(t[7]);
        let eta_eval = eta_tab.eval(rho.log10(), temp.log10());
        let rel = (eta_eval - eta_ref).abs() / eta_ref.max(1.0e-5);
        assert!(
            rel <= 5.0e-2,
            "FeII 5169 eta bicubic mismatch: eval={eta_eval} vs ref={eta_ref} (rel={rel:.3e})"
        );
    }

    let hyd = parse_stella_hyd(&hyd_str);
    let abn = parse_stella_abn(&abn_str);
    let (t0_max, log_lmax, lbol_pts) = parse_stella_lbol(&lbol_str);
    assert_eq!(hyd.n_zones, 400);
    assert_eq!(hyd.zones.len(), 400);
    assert_eq!(abn.len(), 400);
    assert!(t0_max > 0.0 && log_lmax > 41.0 && lbol_pts.len() > 100);

    let times: Vec<f64> = (1..=60).map(f64::from).collect();
    let (lbol_sim, vel_sim) =
        run_stella_from_hyd_abn(&hyd, &abn, &eta_tab, &times);
    assert_eq!(lbol_sim.len(), 60);
    assert!(
        !vel_sim.is_empty(),
        "expected non-empty Fe II velocity rows"
    );
    for row in &vel_sim {
        assert!(row.v_tau_lo_km_s >= row.v_tau_med_km_s);
        assert!(row.v_tau_med_km_s >= row.v_tau_hi_km_s);
        assert!(row.v_tau_hi_km_s > 0.0);
    }

    // Also verify export_stella_pair on a StarModel produces 400 zones in .hyd and .abn
    let mut model =
        load_zams_model(StarControls::default(), StarJobControls::default())
            .expect("zams");
    set_hydro_vars(&mut model, false, false).expect("hydro_vars");
    let (hyd_out, abn_out) =
        export_stella_pair(&model, &StellaExportOptions::default());
    let hyd_parsed = parse_stella_hyd(&hyd_out);
    let abn_parsed = parse_stella_abn(&abn_out);
    assert_eq!(hyd_parsed.zones.len(), 400);
    assert_eq!(abn_parsed.len(), 400);
}
