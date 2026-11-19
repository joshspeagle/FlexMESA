// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary cursor and parser (`FMKAP001` format) for `kap_tables.bin`.

use crate::loader::{
    get_dx_lookup, AesopusData, AesopusSlice, AesopusZSet, KapCnData,
    KapCnZSet, KapCoSet, KapCoSubTable, KapCoXTable, KapCoZTable, KapFixedSet,
    KapTables, KapXTable, KapZTable, KAP_CO_DXS,
};
use crate::types::KapError;
use flexmesa_utils::is_bad;
use std::sync::OnceLock;

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], KapError> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            KapError::TableIo("kap_tables.bin offset overflow".to_string())
        })?;
        let slice = self.bytes.get(self.pos..end).ok_or_else(|| {
            KapError::TableIo("unexpected EOF in kap_tables.bin".to_string())
        })?;
        self.pos = end;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, KapError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<usize, KapError> {
        let s = self.take(4)?;
        let mut buf = [0_u8; 4];
        buf.copy_from_slice(s);
        Ok(u32::from_le_bytes(buf) as usize)
    }
    fn f64(&mut self, label: &str) -> Result<f64, KapError> {
        let s = self.take(8)?;
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(s);
        let val = f64::from_le_bytes(buf);
        if is_bad(val) {
            return Err(KapError::TableIo(format!(
                "non-finite f64 ({val}) in kap_tables.bin for {label}"
            )));
        }
        Ok(val)
    }
    fn f64_vec(&mut self, n: usize, label: &str) -> Result<Vec<f64>, KapError> {
        let total_bytes = n.checked_mul(8).ok_or_else(|| {
            KapError::TableIo(
                "kap_tables.bin vector length overflow".to_string(),
            )
        })?;
        let raw = self.take(total_bytes)?;
        let (chunks, _) = raw.as_chunks::<8>();
        let mut out = Vec::with_capacity(chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            let val = f64::from_le_bytes(*chunk);
            if is_bad(val) {
                return Err(KapError::TableIo(format!(
                    "non-finite f64 ({val}) in kap_tables.bin for {label}[{idx}]"
                )));
            }
            out.push(val);
        }
        Ok(out)
    }
    fn i16_vec(&mut self, n: usize) -> Result<Vec<i16>, KapError> {
        let total_bytes = n.checked_mul(2).ok_or_else(|| {
            KapError::TableIo(
                "kap_tables.bin vector length overflow".to_string(),
            )
        })?;
        let raw = self.take(total_bytes)?;
        let (chunks, _) = raw.as_chunks::<2>();
        Ok(chunks.iter().copied().map(i16::from_le_bytes).collect())
    }
    fn f32_vec(&mut self, n: usize, label: &str) -> Result<Vec<f32>, KapError> {
        let total_bytes = n.checked_mul(4).ok_or_else(|| {
            KapError::TableIo(
                "kap_tables.bin vector length overflow".to_string(),
            )
        })?;
        let raw = self.take(total_bytes)?;
        let (chunks, _) = raw.as_chunks::<4>();
        let mut out = Vec::with_capacity(chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            let val = f32::from_le_bytes(*chunk);
            if !val.is_finite() {
                return Err(KapError::TableIo(format!(
                    "non-finite f32 ({val}) in kap_tables.bin for {label}[{idx}]"
                )));
            }
            out.push(val);
        }
        Ok(out)
    }
    fn string(&mut self) -> Result<String, KapError> {
        let len = self.u32()?;
        let s = std::str::from_utf8(self.take(len)?).map_err(|e| {
            KapError::TableIo(format!("invalid UTF-8 in kap_tables.bin: {e}"))
        })?;
        Ok(s.to_string())
    }
}

/// Unpack the `FMKAP001` binary payload into [`KapTables`].
///
/// # Errors
/// Returns [`KapError::TableIo`] if the binary header, length, finiteness, or
/// payload layout is malformed.
pub fn parse_kap_tables(bytes: &[u8]) -> Result<KapTables, KapError> {
    if bytes.get(..8) != Some(b"FMKAP001".as_slice()) {
        return Err(KapError::TableIo(
            "invalid kap_tables.bin header magic".to_string(),
        ));
    }
    let mut cur = Cursor::new(&bytes[8..]);

    let num_fixed = cur.u32()?;
    let mut fixed_sets = Vec::with_capacity(num_fixed.min(64));
    for _ in 0..num_fixed {
        let prefix = cur.string()?;
        let num_zs = cur.u32()?;
        let mut z_tables = Vec::with_capacity(num_zs.min(64));
        for _ in 0..num_zs {
            let z = cur.f64(&format!("{prefix}.z"))?;
            let num_xs = cur.u32()?;
            let mut x_tables = Vec::with_capacity(num_xs.min(64));
            for _ in 0..num_xs {
                let present = cur.u8()? != 0;
                let x = cur.f64(&format!("{prefix}.x"))?;
                let z_val = cur.f64(&format!("{prefix}.z_val"))?;
                if present {
                    let log_r_min = cur.f64(&format!("{prefix}.log_r_min"))?;
                    let log_r_max = cur.f64(&format!("{prefix}.log_r_max"))?;
                    let log_t_min = cur.f64(&format!("{prefix}.log_t_min"))?;
                    let log_t_max = cur.f64(&format!("{prefix}.log_t_max"))?;
                    let num_lr = cur.u32()?;
                    let num_lt = cur.u32()?;
                    let log_rs =
                        cur.f64_vec(num_lr, &format!("{prefix}.log_rs"))?;
                    let log_ts =
                        cur.f64_vec(num_lt, &format!("{prefix}.log_ts"))?;
                    let grid_len =
                        num_lr.checked_mul(num_lt).ok_or_else(|| {
                            KapError::TableIo(
                                "fixed grid size overflow".to_string(),
                            )
                        })?;
                    let raw_i16 = Some(cur.i16_vec(grid_len)?);
                    x_tables.push(KapXTable {
                        x,
                        z: z_val,
                        log_r_min,
                        log_r_max,
                        log_t_min,
                        log_t_max,
                        log_rs,
                        log_ts,
                        raw_i16,
                        spline: OnceLock::new(),
                    });
                } else {
                    x_tables.push(KapXTable {
                        x,
                        z: z_val,
                        log_r_min: -8.0,
                        log_r_max: 1.0,
                        log_t_min: 2.7,
                        log_t_max: 8.7,
                        log_rs: Vec::new(),
                        log_ts: Vec::new(),
                        raw_i16: None,
                        spline: OnceLock::new(),
                    });
                }
            }
            z_tables.push(KapZTable { z, x_tables });
        }
        fixed_sets.push(KapFixedSet { prefix, z_tables });
    }

    let num_co = cur.u32()?;
    let mut co_sets = Vec::with_capacity(num_co.min(32));
    for _ in 0..num_co {
        let prefix = cur.string()?;
        let num_zs = cur.u32()?;
        let mut z_tables = Vec::with_capacity(num_zs.min(64));
        for _ in 0..num_zs {
            let zbase = cur.f64(&format!("{prefix}.zbase"))?;
            let zfrac_c = cur.f64(&format!("{prefix}.zfrac_c"))?;
            let zfrac_n = cur.f64(&format!("{prefix}.zfrac_n"))?;
            let zfrac_o = cur.f64(&format!("{prefix}.zfrac_o"))?;
            let zfrac_ne = cur.f64(&format!("{prefix}.zfrac_ne"))?;
            let num_xs = cur.u32()?;
            let mut x_tables = Vec::with_capacity(num_xs.min(64));
            for _ in 0..num_xs {
                let present = cur.u8()? != 0;
                let x = cur.f64(&format!("{prefix}.x"))?;
                let z = cur.f64(&format!("{prefix}.z"))?;
                if present {
                    let num_co_tables = cur.u32()?;
                    let num_dxc_gt_dxo = cur.u32()?;
                    let log_r_min = cur.f64(&format!("{prefix}.log_r_min"))?;
                    let log_r_max = cur.f64(&format!("{prefix}.log_r_max"))?;
                    let log_t_min = cur.f64(&format!("{prefix}.log_t_min"))?;
                    let log_t_max = cur.f64(&format!("{prefix}.log_t_max"))?;
                    let num_lr = cur.u32()?;
                    let num_lt = cur.u32()?;
                    let log_rs =
                        cur.f64_vec(num_lr, &format!("{prefix}.log_rs"))?;
                    let log_ts =
                        cur.f64_vec(num_lt, &format!("{prefix}.log_ts"))?;
                    let grid_len =
                        num_lr.checked_mul(num_lt).ok_or_else(|| {
                            KapError::TableIo(
                                "CO grid size overflow".to_string(),
                            )
                        })?;
                    let mut co_table_numbers = [[-1_i32; 8]; 8];
                    let mut co_tables =
                        Vec::with_capacity(num_co_tables.min(128));
                    for n in 0..num_co_tables {
                        let dxc = cur.f64(&format!("{prefix}.dxc"))?;
                        let dxo = cur.f64(&format!("{prefix}.dxo"))?;
                        let keep_sub = cur.u8()? != 0;
                        let raw_i16 = if keep_sub {
                            Some(cur.i16_vec(grid_len)?)
                        } else {
                            None
                        };
                        if let Some(ixc) = KAP_CO_DXS
                            .iter()
                            .position(|&v| (v - dxc).abs() < 1.0e-6)
                        {
                            if let Some(ixo) = KAP_CO_DXS
                                .iter()
                                .position(|&v| (v - dxo).abs() < 1.0e-6)
                            {
                                co_table_numbers[ixc][ixo] = (n + 1) as i32;
                            }
                        }
                        co_tables.push(KapCoSubTable {
                            dxc,
                            dxo,
                            dxc_lookup: get_dx_lookup(dxc, z),
                            dxo_lookup: get_dx_lookup(dxo, z),
                            raw_i16,
                            spline: OnceLock::new(),
                        });
                    }
                    let mut next_dxc_table = vec![-1_i32; num_co_tables];
                    let mut next_dxo_table = vec![-1_i32; num_co_tables];
                    for i in 0..num_co_tables {
                        let dxc = co_tables[i].dxc;
                        let dxo = co_tables[i].dxo;
                        if dxc > dxo {
                            next_dxc_table[i] = (i + 2) as i32;
                        } else {
                            for (j, tj) in co_tables.iter().enumerate() {
                                if tj.dxc > dxc && (tj.dxo - dxo).abs() == 0.0 {
                                    next_dxc_table[i] = (j + 1) as i32;
                                    break;
                                }
                            }
                        }
                        if dxc < dxo {
                            next_dxo_table[i] = (i + 2) as i32;
                        } else {
                            for (j, tj) in co_tables.iter().enumerate() {
                                if tj.dxo > dxo && (tj.dxc - dxc).abs() == 0.0 {
                                    next_dxo_table[i] = (j + 1) as i32;
                                    break;
                                }
                            }
                        }
                    }
                    x_tables.push(KapCoXTable {
                        x,
                        z,
                        present: true,
                        num_co_tables,
                        num_dxc_gt_dxo,
                        log_r_min,
                        log_r_max,
                        log_t_min,
                        log_t_max,
                        log_rs,
                        log_ts,
                        co_table_numbers,
                        next_dxo_table,
                        next_dxc_table,
                        co_tables,
                    });
                } else {
                    x_tables.push(KapCoXTable {
                        x,
                        z,
                        present: false,
                        num_co_tables: 0,
                        num_dxc_gt_dxo: 0,
                        log_r_min: -8.0,
                        log_r_max: 1.0,
                        log_t_min: 3.75,
                        log_t_max: 8.70,
                        log_rs: Vec::new(),
                        log_ts: Vec::new(),
                        co_table_numbers: [[-1; 8]; 8],
                        next_dxo_table: Vec::new(),
                        next_dxc_table: Vec::new(),
                        co_tables: Vec::new(),
                    });
                }
            }
            z_tables.push(KapCoZTable {
                zbase,
                zfrac_c,
                zfrac_n,
                zfrac_o,
                zfrac_ne,
                x_tables,
            });
        }
        co_sets.push(KapCoSet { prefix, z_tables });
    }

    let zsun = cur.f64("aesopus.zsun")?;
    let fco_ref = cur.f64("aesopus.fco_ref")?;
    let fc_ref = cur.f64("aesopus.fc_ref")?;
    let fn_ref = cur.f64("aesopus.fn_ref")?;
    let num_lr = cur.u32()?;
    let num_lt = cur.u32()?;
    let num_zs = cur.u32()?;
    let log_rs = cur.f64_vec(num_lr, "aesopus.log_rs")?;
    let log_ts = cur.f64_vec(num_lt, "aesopus.log_ts")?;
    let zs = cur.f64_vec(num_zs, "aesopus.zs")?;
    let log_zs: Vec<f64> =
        zs.iter().map(|&z| flexmesa_math::log10(z)).collect();
    let aes_grid_len = num_lr.checked_mul(num_lt).ok_or_else(|| {
        KapError::TableIo("AESOPUS grid size overflow".to_string())
    })?;
    let mut ts = Vec::with_capacity(num_zs.min(64));
    for iz in 0..num_zs {
        let n_x = cur.u32()?;
        let n_fco = cur.u32()?;
        let n_fc = cur.u32()?;
        let n_fn = cur.u32()?;
        let xs = cur.f64_vec(n_x, "aesopus.xs")?;
        let fcos = cur.f64_vec(n_fco, "aesopus.fcos")?;
        let fcs = cur.f64_vec(n_fc, "aesopus.fcs")?;
        let fns = cur.f64_vec(n_fn, "aesopus.fns")?;
        let z_present = cur.u8()? != 0;
        let total_slices = n_x
            .checked_mul(n_fco)
            .and_then(|v| v.checked_mul(n_fc))
            .and_then(|v| v.checked_mul(n_fn))
            .ok_or_else(|| {
                KapError::TableIo("AESOPUS slice count overflow".to_string())
            })?;
        let mut slices = Vec::with_capacity(if z_present {
            total_slices.min(16_384)
        } else {
            0
        });
        if z_present {
            for slice_idx in 0..total_slices {
                let sp = cur.u8()? != 0;
                let raw_f32 = if sp {
                    Some(cur.f32_vec(
                        aes_grid_len,
                        &format!("aesopus.z[{iz}].slice[{slice_idx}]"),
                    )?)
                } else {
                    None
                };
                slices.push(AesopusSlice {
                    raw_f32,
                    spline: OnceLock::new(),
                });
            }
        }
        ts.push(AesopusZSet {
            xs,
            fcos,
            fcs,
            fns,
            slices,
        });
    }
    let aesopus = AesopusData {
        zsun,
        fco_ref,
        fc_ref,
        fn_ref,
        log_rs,
        log_ts,
        zs,
        log_zs,
        ts,
    };

    let num_cn_zs = cur.u32()?;
    let cn_num_lr = cur.u32()?;
    let cn_num_lt = cur.u32()?;
    let cn_log_rs = cur.f64_vec(cn_num_lr, "kapcn.log_rs")?;
    let cn_log_ts = cur.f64_vec(cn_num_lt, "kapcn.log_ts")?;
    let cn_grid_len = cn_num_lr.checked_mul(cn_num_lt).ok_or_else(|| {
        KapError::TableIo("kapCN grid size overflow".to_string())
    })?;
    let mut cn_zs = Vec::with_capacity(num_cn_zs.min(64));
    let mut cn_z_sets = Vec::with_capacity(num_cn_zs.min(64));
    for iz in 0..num_cn_zs {
        let z_val = cur.f64(&format!("kapcn.z[{iz}]"))?;
        let mut fcs = [0.0_f64; 7];
        for (k, v) in fcs.iter_mut().enumerate() {
            *v = cur.f64(&format!("kapcn.z[{iz}].fcs[{k}]"))?;
        }
        let mut fns = [0.0_f64; 3];
        for (k, v) in fns.iter_mut().enumerate() {
            *v = cur.f64(&format!("kapcn.z[{iz}].fns[{k}]"))?;
        }
        let mut raw_i16 = Vec::with_capacity(63);
        let mut splines = Vec::with_capacity(63);
        for _ in 0..63 {
            raw_i16.push(cur.i16_vec(cn_grid_len)?);
            splines.push(OnceLock::new());
        }
        cn_zs.push(z_val);
        cn_z_sets.push(KapCnZSet {
            z: z_val,
            fcs,
            fns,
            raw_i16,
            splines,
        });
    }
    if cur.remaining() != 0 {
        return Err(KapError::TableIo(format!(
            "unexpected trailing {} bytes in kap_tables.bin",
            cur.remaining()
        )));
    }
    let kap_cn = KapCnData {
        log_rs: cn_log_rs,
        log_ts: cn_log_ts,
        xs: [0.5, 0.7, 0.8],
        zs: cn_zs,
        z_sets: cn_z_sets,
    };

    Ok(KapTables {
        fixed_sets,
        co_sets,
        aesopus,
        kap_cn,
    })
}

#[cfg(test)]
mod tests {
    use super::parse_kap_tables;
    use crate::types::KapError;

    fn minimal_valid_fmkap001() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"FMKAP001");
        buf.extend_from_slice(&0_u32.to_le_bytes()); // num_fixed
        buf.extend_from_slice(&0_u32.to_le_bytes()); // num_co
                                                     // AESOPUS scalars (zsun, fco_ref, fc_ref, fn_ref) + grid (0, 0, 0)
        for v in [0.0134_f64, 0.0, 0.0, 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        // kapCN grid (0, 0, 0)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf
    }

    #[test]
    fn test_parse_kap_tables_rejects_truncation_trailing_and_nan_without_panic()
    {
        let valid = minimal_valid_fmkap001();
        assert!(parse_kap_tables(&valid).is_ok());

        // Bad magic
        let mut bad_magic = valid.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            parse_kap_tables(&bad_magic),
            Err(KapError::TableIo(_))
        ));

        // Truncated payload
        for cut in [0_usize, 4, 8, valid.len() - 1] {
            assert!(matches!(
                parse_kap_tables(&valid[..cut]),
                Err(KapError::TableIo(_))
            ));
        }

        // Trailing bytes rejected
        let mut trailing = valid.clone();
        trailing.push(0xFF);
        let err = parse_kap_tables(&trailing)
            .err()
            .expect("trailing byte must fail");
        assert!(err.to_string().contains("trailing"), "got: {err}");

        // Non-finite AESOPUS scalar rejected cleanly
        let mut bad_f64 = valid.clone();
        bad_f64[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
        let err = parse_kap_tables(&bad_f64)
            .err()
            .expect("NaN scalar must fail");
        assert!(err.to_string().contains("non-finite f64"), "got: {err}");

        // Corrupted huge length prefix rejected without OOM panic
        let mut huge_len = b"FMKAP001".to_vec();
        huge_len.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            parse_kap_tables(&huge_len),
            Err(KapError::TableIo(_))
        ));
    }
}
