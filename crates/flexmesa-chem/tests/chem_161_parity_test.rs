// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone zero-`$MESA_DIR` MESA `26.4.1` parity tests for `flexmesa-chem`
//! (Issue #161).

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::excessive_precision,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_chem::{
    ag89_yesol, ag89_zsol, builtin_nuclide_rows, category_id, category_name,
    chem_get_element_id, chem_get_solar, chem_m_div_h, chem_xsol,
    element_atomic_weights, element_long_name, element_main_iso_name,
    element_name, eval_composition_moments_generic,
    generate_long_nuclide_names, generate_nuclide_names, generate_nuclide_set,
    get_composition_info, get_composition_info_from_props,
    get_composition_info_zones, get_nuclide_index_in_set,
    get_partition_fcn_indx, get_stable_mass_frac,
    lodders03_element_atom_percent, solar_cno_fraction, solar_element_zfrac,
    ChemError, DualOrder1, NuclideCatalog, SolarScale, SpeciesProps,
    AG89_TABLE, DEL_MN, DEL_MP, EL_LONG_NAME, EL_NAME, MAX_EL_Z, NPART,
    NUM_CATEGORIES, NUM_CHEM_ELEMENTS, PARALLEL_ZONE_THRESHOLD, SOLSIZ, T_PART,
};
use flexmesa_const::{AMU, CLIGHT, FIVE_THIRDS, MEV_TO_ERGS};
use flexmesa_math::{log10, pow_strict};

fn ulp_distance(actual: f64, expected_bits: u64) -> u64 {
    fn signed_magnitude(bits: u64) -> i128 {
        if (bits >> 63) != 0 {
            -((bits ^ (1_u64 << 63)) as i128)
        } else {
            bits as i128
        }
    }
    let a = signed_magnitude(actual.to_bits());
    let b = signed_magnitude(expected_bits);
    u64::try_from((a - b).unsigned_abs()).unwrap_or(u64::MAX)
}

#[test]
fn pinned_mesa_26_4_1_hex_vectors_0_ulp() {
    let cat = NuclideCatalog::builtin_ref();

    // chem/test/src/test_chem.f90:61-87 (`o16` and AG89 bulk constants)
    let o16 = cat.get_by_name("o16").expect("o16 must exist");
    assert_eq!(o16.chem_id, 66);
    assert_eq!(o16.w.to_bits(), 0x402f_fd65_72f7_1260);
    assert_eq!(o16.z, 8);
    assert_eq!(o16.binding_energy.to_bits(), 0x405f_e7a2_a5bc_9298);

    assert_eq!(ag89_zsol().to_bits(), 0x3f93_715b_5ba8_7db0);
    assert_eq!(ag89_yesol().to_bits(), 0x3feb_4a27_3aa6_ab8d);
    let ag89_cno =
        (chem_xsol("c12") + chem_xsol("n14") + chem_xsol("o16")) / ag89_zsol();
    assert_eq!(ag89_cno.to_bits(), 0x3fe7_236d_b531_2df0);

    // chem/test/src/test_chem.f90:102-112 (CNO metal mass fractions)
    assert_eq!(
        solar_cno_fraction(SolarScale::Gn93, cat).to_bits(),
        0x3fe6_bc11_0fbb_f523
    );
    assert_eq!(
        solar_cno_fraction(SolarScale::Gs98, cat).to_bits(),
        0x3fe6_23f1_a5b4_a5ea
    );
    assert_eq!(
        solar_cno_fraction(SolarScale::L03, cat).to_bits(),
        0x3fe5_3781_aeb8_2ae7
    );
    assert_eq!(
        solar_cno_fraction(SolarScale::Ags05, cat).to_bits(),
        0x3fe5_5de0_6651_ac37
    );

    // chem/test/src/test_chem.f90:136-156 (`c12` + `fe56` 50/50 mixture)
    let c12 = cat.get_nuclide_index("c12").expect("c12 present");
    let fe56 = cat.get_nuclide_index("fe56").expect("fe56 present");
    let comp = get_composition_info(cat, &[c12, fe56], &[0.5, 0.5], false)
        .expect("composition_info");
    assert_eq!(ulp_distance(comp.abar, 0x4033_c3c3_c3c3_c3c4), 0);
    assert_eq!(ulp_distance(comp.zbar, 0x4023_0f0f_0f0f_0f0f), 0);
    assert_eq!(ulp_distance(comp.z2bar, 0x4062_9e1e_1e1e_1e1e), 0);
    assert_eq!(ulp_distance(comp.ye, 0x3fde_db6d_b6db_6db6), 0);
    assert_eq!(ulp_distance(comp.mass_correction, 0x3fef_fb3d_bb73_7be6), 0);

    // chem/test/src/test_chem.f90:158-179 (`2 he3 => he4 + 2 h1`)
    let he3 = cat.get_nuclide_index("he3").expect("he3 present");
    let he4 = cat.get_nuclide_index("he4").expect("he4 present");
    let h1 = cat.get_nuclide_index("h1").expect("h1 present");
    let qtotal = cat
        .reaction_qtotal(2, 3, &[he3, he3, he4, h1, h1])
        .expect("reaction_qtotal");
    assert_eq!(qtotal.to_bits(), 0x4029_b819_eebe_8f1f);
}

#[test]
fn composition_info_z2bar_associativity_autodiff_and_zones_parity() {
    let cat = NuclideCatalog::builtin_ref();

    // Left-to-right `sum((y * z) * z) * abar` matches `chem_lib.f90:224`.
    // Reassociated `y * (z * z)` differs by 1 ULP on `c10` (`0x4042000000000000`)
    // and by 2 ULP on `n11` (`0x4048800000000001`).
    let c10 = cat.get_nuclide_index("c10").expect("c10 present");
    let n11 = cat.get_nuclide_index("n11").expect("n11 present");
    let res_c10 = get_composition_info(cat, &[c10], &[1.0], true)
        .expect("c10 composition");
    assert_eq!(res_c10.z2bar.to_bits(), 0x4042_0000_0000_0001);
    let res_n11 = get_composition_info(cat, &[n11], &[1.0], true)
        .expect("n11 composition");
    assert_eq!(res_n11.z2bar.to_bits(), 0x4048_7fff_ffff_ffff);

    // Realistic 8-species mixture spanning H/He burning, AGB, and WD regimes.
    let names = ["h1", "he3", "he4", "c12", "n14", "o16", "ne20", "fe56"];
    let cids: Vec<usize> = names
        .iter()
        .map(|&nm| cat.get_nuclide_index(nm).expect("known isotope"))
        .collect();
    let x = [0.68, 3.0e-5, 0.28, 0.008, 0.003, 0.02, 0.005, 0.00397];
    let props = SpeciesProps::from_catalog(cat, &cids).expect("props");
    let res = get_composition_info_from_props(&props, &x, false);

    // Independent reference calculation of moments & analytical partials.
    let mut sumx = 0.0_f64;
    let mut sumy = 0.0_f64;
    let mut sum_yz = 0.0_f64;
    let mut sum_yz2 = 0.0_f64;
    let mut sum_yz53 = 0.0_f64;
    let mut mc = 0.0_f64;
    for i in 0..8 {
        let yi = x[i] / props.a[i];
        sumx += x[i];
        sumy += yi;
        sum_yz += yi * props.z[i];
        sum_yz2 += yi * props.z[i] * props.z[i];
        sum_yz53 += yi * props.z53[i];
        mc += yi * props.w[i];
    }
    let abar_ref = sumx / sumy;
    assert_eq!(res.abar.to_bits(), abar_ref.to_bits());
    assert_eq!(res.zbar.to_bits(), (sum_yz * abar_ref).to_bits());
    assert_eq!(res.z2bar.to_bits(), (sum_yz2 * abar_ref).to_bits());
    assert_eq!(res.z53bar.to_bits(), (sum_yz53 * abar_ref).to_bits());
    assert_eq!(res.ye.to_bits(), (res.zbar / abar_ref).to_bits());
    assert_eq!(res.mass_correction.to_bits(), mc.to_bits());
    assert_eq!(res.sumx.to_bits(), sumx.to_bits());
    assert!((res.xh - 0.68).abs() < 1e-15);
    assert!((res.xhe - (3.0e-5 + 0.28)).abs() < 1e-15);
    assert!((res.sumx - 1.0).abs() < 1e-15);

    // Verify `DualOrder1<8>` derivatives agree with centered finite differences
    // and with MESA's analytical `chem_lib.f90:238-240` partial definitions.
    let mut x_dual = [DualOrder1::<8>::zero(); 8];
    for i in 0..8 {
        x_dual[i] = DualOrder1::independent(x[i], i);
    }
    let dm = eval_composition_moments_generic(&props, &x_dual);
    assert_eq!(dm.abar.val.to_bits(), res.abar.to_bits());
    assert_eq!(dm.zbar.val.to_bits(), res.zbar.to_bits());
    assert_eq!(dm.z2bar.val.to_bits(), res.z2bar.to_bits());
    assert_eq!(dm.z53bar.val.to_bits(), res.z53bar.to_bits());
    assert_eq!(dm.ye.val.to_bits(), res.ye.to_bits());
    assert_eq!(
        dm.mass_correction.val.to_bits(),
        res.mass_correction.to_bits()
    );
    let h_fd = 1.0e-6_f64;
    for i in 0..8 {
        assert!(
            (dm.abar.d1[i] - res.dabar_dx[i]).abs() < 1e-13,
            "dabar_dx mismatch at {i}"
        );
        assert!(
            (dm.zbar.d1[i] - (res.dzbar_dx[i] + res.zbar / res.sumx)).abs()
                < 1e-13,
            "zbar dual derivative mismatch at {i}"
        );
        assert!(
            (dm.mass_correction.d1[i] - (res.dmc_dx[i] + res.mass_correction))
                .abs()
                < 1e-13,
            "mass_correction dual derivative mismatch at {i}"
        );
        let mut xp = x;
        let mut xm = x;
        xp[i] += h_fd;
        xm[i] -= h_fd;
        let mp = eval_composition_moments_generic::<f64>(&props, &xp);
        let mm = eval_composition_moments_generic::<f64>(&props, &xm);
        let dabar_fd = (mp.abar - mm.abar) / (2.0 * h_fd);
        let dzbar_fd = (mp.zbar - mm.zbar) / (2.0 * h_fd);
        let dmc_fd = (mp.mass_correction - mm.mass_correction) / (2.0 * h_fd);
        assert!(
            (dm.abar.d1[i] - dabar_fd).abs() < 1e-8,
            "FD dabar mismatch at {i}"
        );
        assert!(
            (dm.zbar.d1[i] - dzbar_fd).abs() < 1e-8,
            "FD dzbar mismatch at {i}"
        );
        assert!(
            (dm.mass_correction.d1[i] - dmc_fd).abs() < 1e-8,
            "FD dmc mismatch at {i}"
        );
    }

    // `get_composition_info_zones`: sequential (< 64) and parallel (>= 64).
    for &n_zones in &[17_usize, PARALLEL_ZONE_THRESHOLD + 16] {
        let mut x_zones = Vec::with_capacity(n_zones * 8);
        for z_idx in 0..n_zones {
            let scale = 1.0 + 0.01 * (z_idx as f64);
            for &xi in &x {
                x_zones.push(xi * scale);
            }
        }
        let batch =
            get_composition_info_zones(cat, &cids, &x_zones, n_zones, false)
                .expect("zones batch");
        assert_eq!(batch.len(), n_zones);
        for (z_idx, row) in batch.iter().enumerate() {
            let start = z_idx * 8;
            let single = get_composition_info(
                cat,
                &cids,
                &x_zones[start..start + 8],
                false,
            )
            .expect("single zone");
            assert_eq!(row.abar.to_bits(), single.abar.to_bits());
            assert_eq!(row.zbar.to_bits(), single.zbar.to_bits());
            assert_eq!(row.z2bar.to_bits(), single.z2bar.to_bits());
            assert_eq!(row.z53bar.to_bits(), single.z53bar.to_bits());
            assert_eq!(row.ye.to_bits(), single.ye.to_bits());
            assert_eq!(
                row.mass_correction.to_bits(),
                single.mass_correction.to_bits()
            );
            assert_eq!(row.dabar_dx, single.dabar_dx);
            assert_eq!(row.dzbar_dx, single.dzbar_dx);
            assert_eq!(row.dmc_dx, single.dmc_dx);
        }
    }

    // Error paths on bad lengths / empty / unknown chem_id.
    assert!(matches!(
        get_composition_info(cat, &[], &[], true),
        Err(ChemError::InvalidInput { .. })
    ));
    assert!(matches!(
        get_composition_info(cat, &[c10, n11], &[0.5], true),
        Err(ChemError::InvalidInput { .. })
    ));
    assert!(matches!(
        get_composition_info(cat, &[999_999], &[1.0], true),
        Err(ChemError::InvalidInput { .. })
    ));
    assert!(matches!(
        get_composition_info_zones(cat, &[c10], &[0.5, 0.5], 1, true),
        Err(ChemError::InvalidInput { .. })
    ));
}

#[test]
fn solar_scales_all_ten_normalization_bulk_yz_and_m_div_h() {
    let cat = NuclideCatalog::builtin_ref();
    let weights = element_atomic_weights(cat);
    assert_eq!(weights.len(), NUM_CHEM_ELEMENTS);
    for z in 1..=NUM_CHEM_ELEMENTS {
        let iso = element_main_iso_name(z)
            .and_then(|nm| cat.get_by_name(nm))
            .expect("dominant isotope in catalog");
        assert_eq!(weights[z - 1].to_bits(), iso.w.to_bits());
    }

    let scales = [
        (1, "AG89", SolarScale::Ag89),
        (2, "GN93", SolarScale::Gn93),
        (3, "GS98", SolarScale::Gs98),
        (4, "L03", SolarScale::L03),
        (5, "AGS05", SolarScale::Ags05),
        (6, "AGSS09", SolarScale::Agss09),
        (7, "L09", SolarScale::L09),
        (8, "A09_Prz", SolarScale::A09Prz),
        (9, "MB22_photospheric", SolarScale::Mb22Photospheric),
        (10, "AAG21_photospheric", SolarScale::Aag21Photospheric),
    ];

    for &(id, name, expected) in &scales {
        assert_eq!(SolarScale::from_id(id).expect("valid id"), expected);
        assert_eq!(SolarScale::from_name(name).expect("valid name"), expected);

        let zfrac = solar_element_zfrac(expected, cat);
        match expected {
            SolarScale::Ag89 => {
                let total: f64 = zfrac.iter().sum();
                assert!((total - 1.0).abs() < 1e-14);
            }
            SolarScale::L09 => {
                assert_eq!(zfrac[0], 0.0);
                assert_eq!(zfrac[1], 0.0);
                for &v in &zfrac[92..] {
                    assert_eq!(v, 0.0);
                }
                let total: f64 = zfrac[2..92].iter().sum();
                assert!((total - 1.0).abs() < 1e-14);
            }
            _ => {
                assert_eq!(zfrac[0], -20.0);
                assert_eq!(zfrac[1], -20.0);
                for &v in &zfrac[92..] {
                    assert_eq!(v, -20.0);
                }
                let total: f64 = zfrac[2..92].iter().sum();
                assert!((total - 1.0).abs() < 1e-14);
            }
        }

        // `[M/H]` = 0 at exact `(X_sol, Z_sol)`, +1 at 10 * Z_sol.
        let (ysol, zsol) = expected.bulk_yz();
        let xsol = 1.0 - zsol - ysol;
        let mh_solar = chem_m_div_h(xsol, zsol, expected).expect("solar [M/H]");
        assert_eq!(
            mh_solar.to_bits(),
            (log10(zsol / xsol) - log10(zsol / xsol)).to_bits()
        );
        assert!(mh_solar.abs() < 1e-15);

        let mh_high_z =
            chem_m_div_h(xsol, 10.0 * zsol, expected).expect("10x Z [M/H]");
        assert!((mh_high_z - 1.0).abs() < 1e-14);
        assert_eq!(
            mh_high_z.to_bits(),
            (log10((10.0 * zsol) / xsol) - log10(zsol / xsol)).to_bits()
        );
    }

    assert!(matches!(
        SolarScale::from_id(0),
        Err(ChemError::InvalidInput { .. })
    ));
    assert!(matches!(
        SolarScale::from_id(11),
        Err(ChemError::InvalidInput { .. })
    ));
    assert!(matches!(
        SolarScale::from_name("unknown_scale"),
        Err(ChemError::InvalidInput { .. })
    ));

    for bad_val in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            chem_m_div_h(bad_val, 0.02, SolarScale::Gs98),
            Err(ChemError::InvalidInput { .. })
        ));
        assert!(matches!(
            chem_m_div_h(0.7, bad_val, SolarScale::Gs98),
            Err(ChemError::InvalidInput { .. })
        ));
    }
}

#[test]
fn nuclide_catalog_metadata_bounds_q_and_binding_energy() {
    let cat = NuclideCatalog::builtin_ref();
    assert_eq!(cat.len(), 7856);
    assert!(!cat.is_empty());
    assert_eq!(builtin_nuclide_rows().len(), 7856);
    assert_eq!(cat.del_mp().to_bits(), DEL_MP.to_bits());
    assert_eq!(cat.del_mn().to_bits(), DEL_MN.to_bits());

    assert_eq!(cat.lookup_zn(0, 1), Some(1));
    assert_eq!(cat.lookup_zn(1, 0), Some(2));
    assert_eq!(cat.lookup_zn(8, 8), Some(66));
    let fe56_id = cat.lookup_zn(26, 30).expect("fe56");
    assert_eq!(cat.get_by_id(fe56_id).expect("fe56 entry").name, "fe56");

    let al26_gs = cat.lookup_zn_isomeric_state(13, 13, 0).expect("al26 gs");
    let al26_m1 = cat.lookup_zn_isomeric_state(13, 13, 1).expect("al26-1");
    let al26_m2 = cat.lookup_zn_isomeric_state(13, 13, 2).expect("al26-2");
    assert_ne!(al26_gs, al26_m1);
    assert_ne!(al26_m1, al26_m2);
    assert_eq!(cat.lookup_zn_isomeric_state(13, 13, 3), None);

    // Verified MESA `isotopes.data` per-element `(min_N, max_N)` bounds.
    let expected_bounds = [
        (0_usize, 1_usize, 200_usize),
        (1, 0, 2),
        (2, 1, 8),
        (6, 3, 16),
        (8, 5, 22),
        (13, 8, 38),
        (26, 16, 66),
        (92, 113, 203),
        (112, 150, 224),
    ];
    for &(z, mn, mx) in &expected_bounds {
        assert_eq!(cat.element_min_n(z), Some(mn), "min_N for Z={z}");
        assert_eq!(cat.element_max_n(z), Some(mx), "max_N for Z={z}");
    }
    assert_eq!(cat.element_min_n(MAX_EL_Z + 1), None);
    assert_eq!(cat.element_max_n(MAX_EL_Z + 1), None);

    // Verify `w`, `binding_energy`, `z53`, `get_mass_excess`, `get_q`.
    let o16 = cat.get_by_id(66).expect("o16");
    let conv = MEV_TO_ERGS / AMU / (CLIGHT * CLIGHT);
    assert_eq!(o16.z_plus_n, 16);
    assert_eq!(
        o16.w.to_bits(),
        (16.0_f64 + o16.mass_excess * conv).to_bits()
    );
    assert_eq!(
        o16.binding_energy.to_bits(),
        (8.0 * DEL_MP + 8.0 * DEL_MN - o16.mass_excess).to_bits()
    );
    assert_eq!(
        o16.z53.to_bits(),
        pow_strict(8.0_f64, FIVE_THIRDS).to_bits()
    );

    let mex = cat.get_mass_excess(66).expect("get_mass_excess");
    assert_eq!(cat.get_q(66).expect("get_q").to_bits(), (-mex).to_bits());

    // `NuclideCatalog::binding_energy(Y) = dot_product(binding_energy, Y)`.
    let mut y = vec![0.0_f64; cat.len()];
    let c12_id = cat.get_nuclide_index("c12").expect("c12");
    y[c12_id - 1] = 0.25;
    y[65] = 0.75;
    let c12_b = cat.get_by_id(c12_id).expect("c12").binding_energy;
    let expected_b = c12_b * 0.25 + o16.binding_energy * 0.75;
    assert_eq!(
        cat.binding_energy(&y).expect("binding_energy").to_bits(),
        expected_b.to_bits()
    );

    assert!(matches!(
        cat.binding_energy(&y[..10]),
        Err(ChemError::InvalidInput { .. })
    ));
    assert!(matches!(
        cat.reaction_qtotal(2, 2, &[1, 2]),
        Err(ChemError::InvalidInput { .. })
    ));
}

#[test]
fn isotopes_parser_fortran_d_exponents_trailing_blanks_and_errors() {
    let valid = "\
header line ignored by parser
neut        1.0086649    0    1   0.5   8.071317100
 1.00000D+00  1.00000E+00  1.00000e+00  1.00000d+00  1.00000+00  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
h1          1.0078250    1    0   0.5   7.288970500
 2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0
 2.0  2.0  2.0  2.0  2.0  2.0  2.0  2.0
al26-1     25.9868920   13   13   5.0 -12.210000000
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0

";
    let custom = NuclideCatalog::from_isotopes_str(valid).expect("valid parse");
    assert_eq!(custom.len(), 3);
    assert_eq!(custom.get_by_name("neut").expect("neut").pfcn[4], 1.0);
    let al = custom.get_by_name("al26-1").expect("al26-1");
    assert_eq!(al.isomeric_state, 1);
    assert_eq!(custom.lookup_zn_isomeric_state(13, 13, 1), Some(3));

    // Reordered protons/neutrons resolve `del_mp`/`del_mn` before binding energy.
    let reordered = "\
header
d2          2.0141018    1    1   1.0  13.000000000
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
h1          1.0078250    1    0   0.5   7.500000000
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
neut        1.0086649    0    1   0.5   8.500000000
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
";
    let reord_cat =
        NuclideCatalog::from_isotopes_str(reordered).expect("reordered parse");
    assert_eq!(reord_cat.del_mp(), 7.5);
    assert_eq!(reord_cat.del_mn(), 8.5);
    assert_eq!(
        reord_cat.get_by_name("d2").expect("d2").binding_energy,
        7.5 + 8.5 - 13.0
    );

    // Every malformed variant must fail loudly with `ChemError::TableParse`.
    let bad_inputs = [
        "",
        "only header\n\n",
        "hdr\nh1 1.0 1 0 0.5 7.2\n1 1 1 1 1 1 1 1\n",
        "hdr\nh1 1.0 1 0 0.5\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nh1 badw 1 0 0.5 7.2\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nh1 NaN 1 0 0.5 7.2\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nal26-abc 26.0 13 13 5.0 -12.2\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nal26-100 26.0 13 13 5.0 -12.2\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nal26--1 26.0 13 13 5.0 -12.2\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nh1 1.0 1 0 0.5 7.2\n1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nh1 1.0 1 0 0.5 7.2\n1 1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
        "hdr\nh1 1.0 1 0 0.5 7.2\n1 1 1 foo 1 1 1 1\n1 1 1 1 1 1 1 1\n1 1 1 1 1 1 1 1\n",
    ];
    for (i, &bad) in bad_inputs.iter().enumerate() {
        assert!(
            matches!(
                NuclideCatalog::from_isotopes_str(bad),
                Err(ChemError::TableParse { .. })
            ),
            "bad input {i} must return ChemError::TableParse"
        );
    }
}

#[test]
fn nuclide_set_sort_and_binary_search_parity() {
    assert_eq!(get_nuclide_index_in_set("h1", &[]), None);

    let names = ["o16", "h1", "he4", "c12", "fe56", "al26-1", "al26", "neut"];
    let set = generate_nuclide_set(&names);
    assert_eq!(set.len(), names.len());
    // Must be sorted lexicographically (`al26` precedes `al26-1`).
    for win in set.windows(2) {
        assert!(win[0].nuclide <= win[1].nuclide);
    }

    for (idx, &nm) in names.iter().enumerate() {
        assert_eq!(get_nuclide_index_in_set(nm, &set), Some(idx + 1));
    }
    assert_eq!(get_nuclide_index_in_set("  h1  ", &set), Some(2));
    assert_eq!(get_nuclide_index_in_set("al26", &set), Some(7));
    assert_eq!(get_nuclide_index_in_set("al26-1", &set), Some(6));
    assert_eq!(get_nuclide_index_in_set("a1", &set), None);
    assert_eq!(get_nuclide_index_in_set("h2", &set), None);
    assert_eq!(get_nuclide_index_in_set("zn70", &set), None);
}

#[test]
fn elements_categories_partition_index_and_ag89_decay_parity() {
    assert_eq!(EL_NAME.len(), MAX_EL_Z + 1);
    assert_eq!(EL_LONG_NAME.len(), MAX_EL_Z + 1);
    assert_eq!(element_name(0), Some("neut"));
    assert_eq!(element_name(2), Some("he"));
    assert_eq!(element_long_name(0), Some("neutron"));
    // Literal upstream spelling in `chem_def.f90:109`.
    assert_eq!(EL_LONG_NAME[83], "bisumth");
    assert_eq!(element_long_name(83), Some("bisumth"));

    assert_eq!(chem_get_element_id(" he "), Some(2));
    assert_eq!(chem_get_element_id("FE"), Some(26));
    assert_eq!(chem_get_element_id("neut"), None);
    assert_eq!(chem_get_element_id("unobtainium"), None);

    for id in 1..=NUM_CATEGORIES {
        let name = category_name(id).expect("valid category");
        assert_eq!(category_id(name), id);
    }
    assert_eq!(category_id("  pnhe4 "), 20);
    assert_eq!(category_id("photo"), 21);
    assert_eq!(category_id("other"), 24);
    assert_eq!(category_id("PP"), 0);
    assert_eq!(category_name(0), None);
    assert_eq!(category_name(NUM_CATEGORIES + 1), None);

    // `get_partition_fcn_indx` (`chem_lib.f90:560-593`).
    assert_eq!(get_partition_fcn_indx(0.05), 0);
    assert_eq!(get_partition_fcn_indx(T_PART[0]), 1);
    assert_eq!(get_partition_fcn_indx(0.149), 1);
    assert_eq!(get_partition_fcn_indx(T_PART[1]), 2);
    assert_eq!(get_partition_fcn_indx(9.0), NPART - 1);
    assert_eq!(get_partition_fcn_indx(9.99), NPART - 1);
    assert_eq!(get_partition_fcn_indx(T_PART[NPART - 1]), NPART - 1);
    assert_eq!(get_partition_fcn_indx(25.0), NPART - 1);
    assert_eq!(get_partition_fcn_indx(f64::NAN), NPART - 1);

    // Short & long nuclide name formatting including dual Al-26 isomers.
    let short_single = generate_nuclide_names(&[0, 13, 120], &[1, 26, 300]);
    assert_eq!(short_single, vec!["neut", "al26", "*****"]);
    let short_dual = generate_nuclide_names(&[13, 13], &[26, 26]);
    assert_eq!(short_dual, vec!["al-6", "al*6"]);
    let long_dual =
        generate_long_nuclide_names(&[0, 13, 13, 120], &[1, 26, 26, 300]);
    assert_eq!(
        long_dual,
        vec!["neutron", "Aluminum-gs", "Aluminum-ex", "********"]
    );

    // `chem_get_solar`, `chem_xsol`, `lodders03_element_atom_percent`.
    assert_eq!(SOLSIZ, AG89_TABLE.len());
    assert_eq!(chem_get_solar("p"), chem_get_solar("h1"));
    assert_eq!(chem_get_solar(" d "), chem_get_solar("h2"));
    assert_eq!(chem_get_solar("does_not_exist"), None);
    assert_eq!(lodders03_element_atom_percent("fe56"), 91.754);
    assert_eq!(lodders03_element_atom_percent("does_not_exist"), 0.0);

    // `get_stable_mass_frac`: upstream `be8 -> h2` (`abun_out[1]`, `chem_lib.f90:621`),
    // `jc == 1/2/3`, and odd-odd exceptions (`mn54 -> cr54`, `br80 -> kr80`, `rb84 -> kr84`).
    let idx_of = |nm: &str| {
        AG89_TABLE
            .iter()
            .position(|row| row.0 == nm)
            .expect("AG89 row")
    };
    let be8_decay = get_stable_mass_frac(&[4], &[8], &[1.0]);
    assert_eq!(be8_decay[idx_of("h2")], 1.0);
    assert_eq!(be8_decay[idx_of("he4")], 0.0);

    let k40_decay = get_stable_mass_frac(&[19], &[40], &[1.0]);
    assert_eq!(k40_decay[idx_of("k40")], 1.0);
    assert_eq!(k40_decay[idx_of("ar40")], 0.0);
    assert_eq!(k40_decay[idx_of("ca40")], 0.0);

    let cl40_decay = get_stable_mass_frac(&[17], &[40], &[1.0]);
    assert_eq!(cl40_decay[idx_of("ar40")], 1.0);
    let sc40_decay = get_stable_mass_frac(&[21], &[40], &[1.0]);
    assert_eq!(sc40_decay[idx_of("ca40")], 1.0);

    let mn54_decay = get_stable_mass_frac(&[25], &[54], &[1.0]);
    assert_eq!(mn54_decay[idx_of("cr54")], 1.0);
    assert_eq!(mn54_decay[idx_of("fe54")], 0.0);

    let br80_decay = get_stable_mass_frac(&[35], &[80], &[1.0]);
    assert_eq!(br80_decay[idx_of("kr80")], 1.0);
    assert_eq!(br80_decay[idx_of("se80")], 0.0);

    let rb84_decay = get_stable_mass_frac(&[37], &[84], &[1.0]);
    assert_eq!(rb84_decay[idx_of("kr84")], 1.0);
    assert_eq!(rb84_decay[idx_of("sr84")], 0.0);
}
