// SPDX-License-Identifier: LGPL-3.0-or-later
//! Generator test that extracts sparse bicubic spline patches from an upstream
//! MESA `data` directory (`FLEXMESA_EOS_DATA_DIR`) into
//! `crates/flexmesa-eos/data/eos_tables_4d_patches.bin`.

#![allow(
    clippy::approx_constant,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::excessive_precision,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use flexmesa_eos::{
    locate_log_q, locate_log_t, CmsTableSet, EosDtTableKind, EosDtTableSet,
};
use std::collections::BTreeSet;

#[test]
#[ignore = "Generator test that extracts sparse spline patches from FLEXMESA_EOS_DATA_DIR"]
fn generate_eos_4d_patches_bin() {
    let data_dir = std::env::var("FLEXMESA_EOS_DATA_DIR")
        .expect("FLEXMESA_EOS_DATA_DIR must point to MESA data directory");
    let dt_tables = EosDtTableSet::with_data_dir(&data_dir);
    let cms_tables = CmsTableSet::with_data_dir(&data_dir);

    let mut dt_keys: BTreeSet<(u8, u8, u8, u16, u16)> = BTreeSet::new();
    let mut cms_keys: BTreeSet<(u16, u16, u16)> = BTreeSet::new();

    let record_dt_point = |dt_keys: &mut BTreeSet<(u8, u8, u8, u16, u16)>,
                           kind: EosDtTableKind,
                           ix_range: std::ops::RangeInclusive<usize>,
                           iz_range: std::ops::RangeInclusive<usize>,
                           log_rho: f64,
                           log_t: f64,
                           radius: isize| {
        let log_q = log_rho - 2.0 * log_t + 12.0;
        let k_u8 = match kind {
            EosDtTableKind::OpalScvh => 0u8,
            EosDtTableKind::FreeEos => 1u8,
        };
        for iz in iz_range {
            let max_ix = match kind {
                EosDtTableKind::OpalScvh => {
                    dt_tables.opal_xz.xs_for_z[iz - 1].len()
                }
                EosDtTableKind::FreeEos => {
                    dt_tables.free_xz.xs_for_z[iz - 1].len()
                }
            };
            for ix in ix_range.clone() {
                if ix > max_ix {
                    continue;
                }
                if let Ok(tbl) = dt_tables.get_table(kind, ix, iz) {
                    if let (Ok((iq, _, _, _)), Ok((jt, _, _, _))) = (
                        locate_log_q(&tbl, log_q, true),
                        locate_log_t(&tbl, log_t, true),
                    ) {
                        for dq in -radius..=radius + 1 {
                            for dt in -radius..=radius + 1 {
                                let q_idx = (iq as isize + dq)
                                    .clamp(1, tbl.num_log_qs as isize)
                                    as u16;
                                let t_idx = (jt as isize + dt)
                                    .clamp(1, tbl.num_log_ts as isize)
                                    as u16;
                                dt_keys.insert((
                                    k_u8, ix as u8, iz as u8, q_idx, t_idx,
                                ));
                            }
                        }
                    }
                }
            }
        }
    };

    let doit_trho = [
        (1.0e6_f64, 1.0e-2_f64),
        (1.0e4, 1.0e-1),
        (1.0e5, 1.0e-1),
        (2.0e8, 1.0e2),
        (1.0e5, 1.0e-2),
        (1.0e6, 3.162_277_660_168_379_5e3),
    ];

    for &(t, rho) in &doit_trho {
        let lt = t.log10();
        let lr = rho.log10();
        record_dt_point(
            &mut dt_keys,
            EosDtTableKind::OpalScvh,
            1..=6,
            1..=3,
            lr,
            lt,
            0,
        );
        record_dt_point(
            &mut dt_keys,
            EosDtTableKind::FreeEos,
            1..=11,
            1..=3,
            lr,
            lt,
            0,
        );
    }

    for (kind, k_u8, ix_range) in [
        (EosDtTableKind::OpalScvh, 0u8, 3..=5usize),
        (EosDtTableKind::FreeEos, 1u8, 6..=10usize),
    ] {
        for iz in 2..=3usize {
            for ix in ix_range.clone() {
                if let Ok(tbl) = dt_tables.get_table(kind, ix, iz) {
                    for &(lt, lq_lo, lq_hi) in
                        &[(4.8_f64, -5.6_f64, -4.0_f64), (3.0, -2.5, 2.5)]
                    {
                        if let (
                            Ok((jt, _, _, _)),
                            Ok((iq0, _, _, _)),
                            Ok((iq1, _, _, _)),
                        ) = (
                            locate_log_t(&tbl, lt, true),
                            locate_log_q(&tbl, lq_lo, true),
                            locate_log_q(&tbl, lq_hi, true),
                        ) {
                            for iq in iq0.saturating_sub(2).max(1)
                                ..=(iq1 + 3).min(tbl.num_log_qs)
                            {
                                for dt in 0..=1 {
                                    let t_idx =
                                        (jt + dt).min(tbl.num_log_ts) as u16;
                                    dt_keys.insert((
                                        k_u8, ix as u8, iz as u8, iq as u16,
                                        t_idx,
                                    ));
                                }
                            }
                        }
                    }
                    for jt in 110..=165usize.min(tbl.num_log_ts - 1) {
                        let lt0 =
                            tbl.log_t_min + (jt - 1) as f64 * tbl.del_log_t;
                        let lt1 = tbl.log_t_min + (jt as f64) * tbl.del_log_t;
                        let lq0 = -7.0 - 2.0 * lt0 + 12.0;
                        let lq1 = -7.0 - 2.0 * lt1 + 12.0;
                        if let (Ok((iq0, _, _, _)), Ok((iq1, _, _, _))) = (
                            locate_log_q(&tbl, lq0, true),
                            locate_log_q(&tbl, lq1, true),
                        ) {
                            let iq_min = iq0.min(iq1).saturating_sub(1).max(1);
                            let iq_max = (iq0.max(iq1) + 2).min(tbl.num_log_qs);
                            for iq in iq_min..=iq_max {
                                dt_keys.insert((
                                    k_u8, ix as u8, iz as u8, iq as u16,
                                    jt as u16,
                                ));
                                dt_keys.insert((
                                    k_u8,
                                    ix as u8,
                                    iz as u8,
                                    iq as u16,
                                    (jt + 1) as u16,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    for ix in 6..=10 {
        if let Ok(tbl) = cms_tables.get_table(ix) {
            for it in 58..=64 {
                for ir in 118..=124 {
                    if tbl.get_vertex(it, ir).is_some() {
                        cms_keys.insert((ix as u16, it as u16, ir as u16));
                    }
                }
            }
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(dt_keys.len() as u32).to_le_bytes());
    for &(k_u8, ix, iz, iq, jt) in &dt_keys {
        let kind = if k_u8 == 0 {
            EosDtTableKind::OpalScvh
        } else {
            EosDtTableKind::FreeEos
        };
        let tbl = dt_tables
            .get_table(kind, ix as usize, iz as usize)
            .expect("loaded table");
        let v = tbl
            .get_vertex(iq as usize, jt as usize)
            .expect("valid vertex");
        bytes.extend_from_slice(&[k_u8, ix, iz, 0]);
        bytes.extend_from_slice(&iq.to_le_bytes());
        bytes.extend_from_slice(&jt.to_le_bytes());
        for c in &v.coeffs {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }

    bytes.extend_from_slice(&(cms_keys.len() as u32).to_le_bytes());
    for &(ix, it, ir) in &cms_keys {
        let tbl = cms_tables.get_table(ix as usize).expect("loaded CMS table");
        let v = tbl
            .get_vertex(it as usize, ir as usize)
            .expect("valid CMS vertex");
        bytes.extend_from_slice(&ix.to_le_bytes());
        bytes.extend_from_slice(&it.to_le_bytes());
        bytes.extend_from_slice(&ir.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        for c in &v.coeffs {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
    }

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/eos_tables_4d_patches.bin");
    std::fs::write(&out_path, &bytes).expect("write eos_tables_4d_patches.bin");
}
