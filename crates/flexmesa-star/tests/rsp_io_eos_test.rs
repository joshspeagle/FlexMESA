// SPDX-License-Identifier: LGPL-3.0-or-later
//! Integration tests for `rsp_config`, `rsp_def`, `rsp_eos_kap`, `rsp_eos_inv`,
//! and lossless `32812` (`RSP`) / `RSP2` `.mod` and `.photo` serialization.

use std::path::PathBuf;

use flexmesa_kap::{KapCoOption, KapLowTOption, KapOption};
use flexmesa_star::{
    compute_photo_checksum, eval1_gamma_pt_get_rho, eval1_mesa_eos_de_and_kap,
    eval1_mesa_eos_degas_and_kap, eval1_mesa_rho_given_pt,
    eval1_mesa_t_given_dp, eval_mesa_eos_and_kap, get_or_load_net,
    get_surf_p_t_kap, read_mod_str, read_photo_bytes, set_rho_for_new_pgas,
    set_t_for_new_egas, set_t_for_new_energy, set_t_for_new_pgas,
    write_mod_string, write_photo_bytes, InlistBundle, RspChemCache,
    RspDerivedConsts, StarControls, StarJobControls,
};

fn rsp_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/golden/rsp")
}

/// Verifies that `RspChemCache::from_controls` matches the 10-isotope
/// `o18_and_ne22.net` mass fractions in `rsp_cepheid_cycle0.mod` to 14+ digits.
#[test]
fn test_rsp_standard_metals_xa_matches_cepheid_cycle0_to_15_digits() {
    let mod_text = std::fs::read_to_string(
        rsp_fixture_dir().join("rsp_cepheid_cycle0.mod"),
    )
    .expect("read rsp_cepheid_cycle0.mod");
    let model = read_mod_str(
        &mod_text,
        StarControls::default(),
        StarJobControls::default(),
    )
    .expect("parse rsp_cepheid_cycle0.mod");

    let net = get_or_load_net("o18_and_ne22.net", None).expect("o18_and_ne22");
    assert_eq!(net.chem_id, model.chem_id);

    let job = StarJobControls {
        initial_zfracs: 6, // AGSS09 (`inlist_rsp_common`)
        ..StarJobControls::default()
    };
    let controls = StarControls {
        rsp_x: 0.73,
        rsp_z: 0.007,
        ..StarControls::default()
    };

    let chem =
        RspChemCache::from_controls(&net.chem_id, &job, &controls).expect("xa");
    let xa_golden = model.cell_xa(0);
    for (j, (&got, &want)) in chem.xa.iter().zip(xa_golden.iter()).enumerate() {
        let rel = (got - want).abs() / want.abs().max(1.0e-30);
        assert!(
            rel <= 5.0e-14,
            "isotope {} ({}): got={got:.17e}, want={want:.17e}, rel={rel:.3e}",
            j,
            model.iso_names[j]
        );
    }
}

/// Verifies lossless `.mod` (`32812`) and `.photo` round-tripping across all 7
/// upstream `RSP` cycle-0 golden `.mod` fixtures.
#[test]
fn test_all_seven_rsp_golden_mods_and_photos_roundtrip_losslessly() {
    let fixtures = [
        "rsp_cepheid_cycle0.mod",
        "rsp_cepheid_6M_cycle0.mod",
        "rsp_RR_Lyrae_cycle0.mod",
        "rsp_Delta_Scuti_cycle0.mod",
        "rsp_BEP_cycle0.mod",
        "rsp_BLAP_cycle0.mod",
        "rsp_Type_II_Cepheid_cycle0.mod",
    ];
    for fname in fixtures {
        let raw = std::fs::read_to_string(rsp_fixture_dir().join(fname))
            .unwrap_or_else(|e| panic!("read {fname}: {e}"));
        let mut m1 = read_mod_str(
            &raw,
            StarControls::default(),
            StarJobControls::default(),
        )
        .unwrap_or_else(|e| panic!("parse {fname}: {e}"));
        assert!(m1.controls.rsp_flag && m1.controls.v_flag);

        let serialized = write_mod_string(&m1).expect("write_mod_string");
        assert!(serialized.contains("         32812 -- model for mesa/star"));
        let m2 = read_mod_str(
            &serialized,
            StarControls::default(),
            StarJobControls::default(),
        )
        .expect("re-parse serialized .mod");

        assert_eq!((m2.nz, m2.species), (m1.nz, m1.species));
        assert!((m2.xmstar - m1.xmstar).abs() <= 1.0e-15 * m1.xmstar);
        assert!((m2.r_center - m1.r_center).abs() <= 1.0e-15 * m1.r_center);
        assert!((m2.l_center - m1.l_center).abs() <= 1.0e-15 * m1.l_center);
        assert!((m2.dt - m1.dt).abs() <= 1.0e-15 * m1.dt);
        for k in 0..m1.nz {
            assert!(
                (m2.lnd[k] - m1.lnd[k]).abs()
                    <= 1.0e-15 * m1.lnd[k].abs().max(1.0)
            );
            assert!(
                (m2.ln_t[k] - m1.ln_t[k]).abs()
                    <= 1.0e-15 * m1.ln_t[k].abs().max(1.0)
            );
            assert!((m2.dq[k] - m1.dq[k]).abs() <= 1.0e-15 * m1.dq[k]);
            assert!(
                (m2.aux.rsp.et_rsp[k] - m1.aux.rsp.et_rsp[k]).abs()
                    <= 1.0e-15 * m1.aux.rsp.et_rsp[k].abs().max(1.0e-30)
            );
            assert!(
                (m2.aux.rsp.erad_rsp[k] - m1.aux.rsp.erad_rsp[k]).abs()
                    <= 1.0e-15 * m1.aux.rsp.erad_rsp[k].abs().max(1.0)
            );
            assert!(
                (m2.aux.rsp.fr_rsp[k] - m1.aux.rsp.fr_rsp[k]).abs()
                    <= 1.0e-15 * m1.aux.rsp.fr_rsp[k].abs().max(1.0)
            );
        }

        m1.aux.rsp.num_periods = 3;
        m1.aux.rsp.period = 4.123_456_789e5;
        m1.aux.rsp.have_set_velocities = true;
        m1.aux.rsp.work[10] = 9.876_543_21e35;
        let photo1 = write_photo_bytes(&m1);
        let restored = read_photo_bytes(
            &photo1,
            StarControls::default(),
            StarJobControls::default(),
        )
        .expect("read_photo_bytes");
        let photo2 = write_photo_bytes(&restored);
        assert_eq!(
            compute_photo_checksum(&photo1),
            compute_photo_checksum(&photo2),
            "{fname} .photo round-trip checksum mismatch"
        );
        assert_eq!(restored.aux.rsp.num_periods, 3);
        assert!((restored.aux.rsp.work[10] - 9.876_543_21e35).abs() <= 1.0e-15);
    }
}

/// Verifies `RSP2` (`w`, `Hp`) `.mod` columns and `RSP` / `RSP2` inlist parsing.
#[test]
fn test_rsp2_mod_columns_and_inlist_namelist_roundtrip() {
    let raw = std::fs::read_to_string(
        rsp_fixture_dir().join("rsp_cepheid_cycle0.mod"),
    )
    .expect("read cepheid");
    let mut rsp2_mod =
        read_mod_str(&raw, StarControls::default(), StarJobControls::default())
            .expect("parse");
    rsp2_mod.controls.rsp_flag = false;
    rsp2_mod.controls.rsp2_flag = true;
    for k in 0..rsp2_mod.nz {
        let idx = f64::from(u32::try_from(k).unwrap_or(0));
        let w = 1.0e4 * (idx + 1.0);
        rsp2_mod.aux.etrb[k] = w * w;
        rsp2_mod.aux.rsp.hp_face[k] = 5.0e10 + 1.0e8 * idx;
    }
    let rsp2_str = write_mod_string(&rsp2_mod).expect("write RSP2 .mod");
    let rsp2_back = read_mod_str(
        &rsp2_str,
        StarControls::default(),
        StarJobControls::default(),
    )
    .expect("read RSP2 .mod");
    assert!(rsp2_back.controls.rsp2_flag && !rsp2_back.controls.rsp_flag);
    for k in [0, rsp2_mod.nz / 2, rsp2_mod.nz - 1] {
        assert!(
            (rsp2_back.aux.etrb[k] - rsp2_mod.aux.etrb[k]).abs()
                <= 1.0e-13 * rsp2_mod.aux.etrb[k]
        );
        assert!(
            (rsp2_back.aux.rsp.hp_face[k] - rsp2_mod.aux.rsp.hp_face[k]).abs()
                <= 1.0e-14 * rsp2_mod.aux.rsp.hp_face[k]
        );
    }

    let inlist_str = "&star_job\n  change_RSP2_flag = .true.\n  new_RSP2_flag = .true.\n/\n&controls\n  RSP_nz = 200\n  RSP_nz_outer = 60\n  RSP_alfap = 0.5d0\n  RSP2_alfad = 1.25d0\n/\n";
    let tmp_path = std::env::temp_dir().join(format!(
        "flexmesa_rsp_inlist_test_{}.namelist",
        std::process::id()
    ));
    std::fs::write(&tmp_path, inlist_str).expect("write temp inlist");
    let bundle = InlistBundle::from_file(&tmp_path).expect("inlist");
    let _ = std::fs::remove_file(&tmp_path);
    assert!(bundle.controls.rsp2_flag);
    assert_eq!(
        (bundle.controls.rsp.nz, bundle.controls.rsp.nz_outer),
        (200, 60)
    );
    assert!((bundle.controls.rsp.alfap - 0.5).abs() <= 1.0e-15);
    assert!((bundle.controls.rsp2.alfad - 1.25).abs() <= 1.0e-15);
}

/// Verifies analytical $(V, T)$ partials against centered finite differences
/// and `get_surf_p_t_kap` boundary convergence.
#[test]
fn test_rsp_eos_kap_jacobians_and_surf_bc() {
    let raw = std::fs::read_to_string(
        rsp_fixture_dir().join("rsp_cepheid_cycle0.mod"),
    )
    .expect("read cepheid");
    let mut model =
        read_mod_str(&raw, StarControls::default(), StarJobControls::default())
            .expect("parse cepheid");
    model.kap_config.zbase = 0.007;
    model.kap_config.kap_option = KapOption::A09;
    model.kap_config.kap_co_option = KapCoOption::A09Co;
    model.kap_config.kap_low_t_option = KapLowTOption::LowTFa05A09p;

    let chem = RspChemCache::from_model_zone0(&model).expect("chem cache");
    let derived = RspDerivedConsts::from_config(&model.controls.rsp, model.nz);
    assert_eq!(derived.ibotom, 5);
    assert!((derived.alfa - 1.5).abs() <= 1.0e-15);

    for &k in &[5_usize, 45, 110] {
        let t0 = model.ln_t[k].exp();
        let v0 = 1.0 / model.lnd[k].exp();
        let base = eval_mesa_eos_and_kap(&mut model, &chem, Some(k), t0, v0)
            .expect("base eos_kap");

        let dt = 1.0e-6 * t0;
        let t_hi = eval_mesa_eos_and_kap(&mut model, &chem, None, t0 + dt, v0)
            .unwrap();
        let t_lo = eval_mesa_eos_and_kap(&mut model, &chem, None, t0 - dt, v0)
            .unwrap();
        let dpgas_temp_fd = (t_hi.pgas - t_lo.pgas) / (2.0 * dt);
        let denergy_temp_fd = (t_hi.egas - t_lo.egas) / (2.0 * dt);
        let dkap_temp_fd = (t_hi.op - t_lo.op) / (2.0 * dt);
        assert!(
            (base.d_pg_dt - dpgas_temp_fd).abs() <= 1.0e-5 * base.d_pg_dt.abs()
        );
        assert!(
            (base.d_egas_dt - denergy_temp_fd).abs()
                <= 1.0e-5 * base.d_egas_dt.abs()
        );
        assert!(
            (base.opt - dkap_temp_fd).abs()
                <= 2.0e-4 * base.opt.abs().max(base.op / t0)
        );

        let dv = 1.0e-6 * v0;
        let v_hi = eval_mesa_eos_and_kap(&mut model, &chem, None, t0, v0 + dv)
            .unwrap();
        let v_lo = eval_mesa_eos_and_kap(&mut model, &chem, None, t0, v0 - dv)
            .unwrap();
        let dpgas_vol_fd = (v_hi.pgas - v_lo.pgas) / (2.0 * dv);
        let denergy_vol_fd = (v_hi.egas - v_lo.egas) / (2.0 * dv);
        assert!(
            (base.d_pg_dv - dpgas_vol_fd).abs() <= 1.0e-5 * base.d_pg_dv.abs()
        );
        assert!(
            (base.d_egas_dv - denergy_vol_fd).abs()
                <= 1.0e-4 * base.d_egas_dv.abs().max(base.egas / v0)
        );
    }

    let r0 = model.ln_r[0].exp();
    let (t_surf, p_surf, kap_surf, teff) = get_surf_p_t_kap(
        &model,
        &chem,
        model.mstar,
        r0,
        model.lum[0],
        0.0015,
        1.0e-2,
    )
    .expect("get_surf_p_t_kap");
    let teff_surf_exact = (model.lum[0]
        / (4.0 * std::f64::consts::PI * r0 * r0 * flexmesa_const::BOLTZ_SIGMA))
        .powf(0.25);
    assert!(
        t_surf > 4000.0 && t_surf < 7000.0 && p_surf > 0.0 && kap_surf > 0.0
    );
    assert!((teff - teff_surf_exact).abs() <= 1.0e-11 * teff_surf_exact);
    assert!((teff - model.teff).abs() <= 30.0);
}

/// Verifies all 8 inverse EOS root-finders converge to `< 1e-9`..`< 1e-10`.
#[test]
fn test_rsp_inverse_eos_root_finders_converge() {
    let raw = std::fs::read_to_string(
        rsp_fixture_dir().join("rsp_cepheid_cycle0.mod"),
    )
    .expect("read cepheid");
    let mut model =
        read_mod_str(&raw, StarControls::default(), StarJobControls::default())
            .expect("parse cepheid");
    let chem = RspChemCache::from_model_zone0(&model).expect("chem cache");

    for &k in &[5_usize, 45, 110] {
        let t0 = model.ln_t[k].exp();
        let v0 = 1.0 / model.lnd[k].exp();
        let base = eval_mesa_eos_and_kap(&mut model, &chem, Some(k), t0, v0)
            .expect("base eos_kap");
        let ptot = base.pgas + base.prad;

        let rho_guess = eval1_gamma_pt_get_rho(&chem, ptot, t0);
        let rho_inv =
            eval1_mesa_rho_given_pt(&model, &chem, ptot, t0, rho_guess)
                .expect("rho_given_pt");
        assert!(((1.0 / rho_inv) - v0).abs() <= 1.0e-9 * v0);

        let temp_from_p =
            eval1_mesa_t_given_dp(&model, &chem, v0, ptot, t0 * 0.97)
                .expect("t_given_dp");
        assert!((temp_from_p - t0).abs() <= 1.0e-10 * t0);

        model.t[k] = t0 * 1.02;
        let (temp_from_egas, _) = eval1_mesa_eos_degas_and_kap(
            &mut model, &chem, k, true, base.egas, v0,
        )
        .expect("eos_degas");
        assert!((temp_from_egas - t0).abs() <= 1.0e-10 * t0);

        model.t[k] = t0 * 0.98;
        let (temp_from_etot, _) = eval1_mesa_eos_de_and_kap(
            &mut model,
            &chem,
            k,
            true,
            base.egas + base.erad,
            v0,
        )
        .expect("eos_de");
        assert!((temp_from_etot - t0).abs() <= 1.0e-10 * t0);

        model.pgas[k] = base.pgas;
        model.rho[k] = (1.0 / v0) * 1.03;
        model.ln_t[k] = t0.ln();
        set_rho_for_new_pgas(&mut model, &chem, k).expect("rho_for_pgas");
        assert!((model.rho[k] - 1.0 / v0).abs() <= 1.0e-10 / v0);

        model.lnd[k] = (1.0 / v0).ln();
        model.rho[k] = 1.0 / v0;
        model.aux.rsp.egas_rsp[k] = base.egas;
        model.t[k] = t0 * 1.02;
        set_t_for_new_egas(&mut model, &chem, k).expect("t_for_egas");
        assert!((model.t[k] - t0).abs() <= 1.0e-10 * t0);

        model.t[k] = t0 * 0.98;
        set_t_for_new_pgas(&mut model, &chem, k).expect("t_for_pgas");
        assert!((model.t[k] - t0).abs() <= 1.0e-10 * t0);

        model.energy[k] = base.egas + base.erad;
        model.t[k] = t0 * 1.02;
        set_t_for_new_energy(&mut model, &chem, k, 1.0e-11, 1.0e-11)
            .expect("t_for_energy");
        assert!((model.t[k] - t0).abs() <= 1.0e-10 * t0);
    }
}
