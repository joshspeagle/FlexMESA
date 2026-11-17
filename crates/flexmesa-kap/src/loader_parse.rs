// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary cursor and parser (`FMKAP001` format) for `kap_tables.bin`.

use crate::loader::{
    get_dx_lookup, AesopusData, AesopusSlice, AesopusZSet, KapCnData,
    KapCnZSet, KapCoSet, KapCoSubTable, KapCoXTable, KapCoZTable, KapFixedSet,
    KapTables, KapXTable, KapZTable, KAP_CO_DXS,
};
use crate::types::KapError;
use std::sync::OnceLock;

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
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
    fn f64(&mut self) -> Result<f64, KapError> {
        let s = self.take(8)?;
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(s);
        Ok(f64::from_le_bytes(buf))
    }
    fn f64_vec(&mut self, n: usize) -> Result<Vec<f64>, KapError> {
        let total_bytes = n.checked_mul(8).ok_or_else(|| {
            KapError::TableIo(
                "kap_tables.bin vector length overflow".to_string(),
            )
        })?;
        let raw = self.take(total_bytes)?;
        let (chunks, _) = raw.as_chunks::<8>();
        Ok(chunks.iter().copied().map(f64::from_le_bytes).collect())
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
    fn f32_vec(&mut self, n: usize) -> Result<Vec<f32>, KapError> {
        let total_bytes = n.checked_mul(4).ok_or_else(|| {
            KapError::TableIo(
                "kap_tables.bin vector length overflow".to_string(),
            )
        })?;
        let raw = self.take(total_bytes)?;
        let (chunks, _) = raw.as_chunks::<4>();
        Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
    }
    fn string(&mut self) -> Result<String, KapError> {
        let len = self.u32()?;
        let s = std::str::from_utf8(self.take(len)?).map_err(|e| {
            KapError::TableIo(format!("invalid UTF-8 in kap_tables.bin: {e}"))
        })?;
        Ok(s.to_string())
    }
}

/// Unpack the embedded `kap_tables.bin` payload into [`KapTables`].
///
/// # Errors
/// Returns [`KapError::TableIo`] if the binary header or payload is malformed.
pub(crate) fn parse_kap_tables(bytes: &[u8]) -> Result<KapTables, KapError> {
    if bytes.get(..8) != Some(b"FMKAP001".as_slice()) {
        return Err(KapError::TableIo(
            "invalid kap_tables.bin header magic".to_string(),
        ));
    }
    let mut cur = Cursor::new(&bytes[8..]);

    let num_fixed = cur.u32()?;
    let mut fixed_sets = Vec::with_capacity(num_fixed);
    for _ in 0..num_fixed {
        let prefix = cur.string()?;
        let num_zs = cur.u32()?;
        let mut z_tables = Vec::with_capacity(num_zs);
        for _ in 0..num_zs {
            let z = cur.f64()?;
            let num_xs = cur.u32()?;
            let mut x_tables = Vec::with_capacity(num_xs);
            for _ in 0..num_xs {
                let present = cur.u8()? != 0;
                let x = cur.f64()?;
                let z_val = cur.f64()?;
                if present {
                    let log_r_min = cur.f64()?;
                    let log_r_max = cur.f64()?;
                    let log_t_min = cur.f64()?;
                    let log_t_max = cur.f64()?;
                    let num_lr = cur.u32()?;
                    let num_lt = cur.u32()?;
                    let log_rs = cur.f64_vec(num_lr)?;
                    let log_ts = cur.f64_vec(num_lt)?;
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
    let mut co_sets = Vec::with_capacity(num_co);
    for _ in 0..num_co {
        let prefix = cur.string()?;
        let num_zs = cur.u32()?;
        let mut z_tables = Vec::with_capacity(num_zs);
        for _ in 0..num_zs {
            let zbase = cur.f64()?;
            let zfrac_c = cur.f64()?;
            let zfrac_n = cur.f64()?;
            let zfrac_o = cur.f64()?;
            let zfrac_ne = cur.f64()?;
            let num_xs = cur.u32()?;
            let mut x_tables = Vec::with_capacity(num_xs);
            for _ in 0..num_xs {
                let present = cur.u8()? != 0;
                let x = cur.f64()?;
                let z = cur.f64()?;
                if present {
                    let num_co_tables = cur.u32()?;
                    let num_dxc_gt_dxo = cur.u32()?;
                    let log_r_min = cur.f64()?;
                    let log_r_max = cur.f64()?;
                    let log_t_min = cur.f64()?;
                    let log_t_max = cur.f64()?;
                    let num_lr = cur.u32()?;
                    let num_lt = cur.u32()?;
                    let log_rs = cur.f64_vec(num_lr)?;
                    let log_ts = cur.f64_vec(num_lt)?;
                    let grid_len =
                        num_lr.checked_mul(num_lt).ok_or_else(|| {
                            KapError::TableIo(
                                "CO grid size overflow".to_string(),
                            )
                        })?;
                    let mut co_table_numbers = [[-1_i32; 8]; 8];
                    let mut co_tables = Vec::with_capacity(num_co_tables);
                    for n in 0..num_co_tables {
                        let dxc = cur.f64()?;
                        let dxo = cur.f64()?;
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

    let zsun = cur.f64()?;
    let fco_ref = cur.f64()?;
    let fc_ref = cur.f64()?;
    let fn_ref = cur.f64()?;
    let num_lr = cur.u32()?;
    let num_lt = cur.u32()?;
    let num_zs = cur.u32()?;
    let log_rs = cur.f64_vec(num_lr)?;
    let log_ts = cur.f64_vec(num_lt)?;
    let zs = cur.f64_vec(num_zs)?;
    let log_zs: Vec<f64> =
        zs.iter().map(|&z| flexmesa_math::log10(z)).collect();
    let aes_grid_len = num_lr.checked_mul(num_lt).ok_or_else(|| {
        KapError::TableIo("AESOPUS grid size overflow".to_string())
    })?;
    let mut ts = Vec::with_capacity(num_zs);
    for _ in 0..num_zs {
        let n_x = cur.u32()?;
        let n_fco = cur.u32()?;
        let n_fc = cur.u32()?;
        let n_fn = cur.u32()?;
        let xs = cur.f64_vec(n_x)?;
        let fcos = cur.f64_vec(n_fco)?;
        let fcs = cur.f64_vec(n_fc)?;
        let fns = cur.f64_vec(n_fn)?;
        let z_present = cur.u8()? != 0;
        let total_slices = n_x
            .checked_mul(n_fco)
            .and_then(|v| v.checked_mul(n_fc))
            .and_then(|v| v.checked_mul(n_fn))
            .ok_or_else(|| {
                KapError::TableIo("AESOPUS slice count overflow".to_string())
            })?;
        let mut slices =
            Vec::with_capacity(if z_present { total_slices } else { 0 });
        if z_present {
            for _ in 0..total_slices {
                let sp = cur.u8()? != 0;
                let raw_f32 = if sp {
                    Some(cur.f32_vec(aes_grid_len)?)
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
    let cn_log_rs = cur.f64_vec(cn_num_lr)?;
    let cn_log_ts = cur.f64_vec(cn_num_lt)?;
    let cn_grid_len = cn_num_lr.checked_mul(cn_num_lt).ok_or_else(|| {
        KapError::TableIo("kapCN grid size overflow".to_string())
    })?;
    let mut cn_zs = Vec::with_capacity(num_cn_zs);
    let mut cn_z_sets = Vec::with_capacity(num_cn_zs);
    for _ in 0..num_cn_zs {
        let z_val = cur.f64()?;
        let mut fcs = [0.0_f64; 7];
        for v in &mut fcs {
            *v = cur.f64()?;
        }
        let mut fns = [0.0_f64; 3];
        for v in &mut fns {
            *v = cur.f64()?;
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
