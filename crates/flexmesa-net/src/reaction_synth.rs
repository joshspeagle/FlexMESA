// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dynamic REACLIB reaction handle parser and [`ReactionInfo`] synthesizer
//! (`rates/private/reaclib_support.f90` and `rates/private/rates_initialize.f90`).

use flexmesa_chem::{category_id, NuclideCatalog};
use flexmesa_const::LN2;
use flexmesa_rates::{eval_reaclib_handle, RatesCatalog, ReactionInfo};

use crate::types::NetError;

/// Parses a reaction handle `r_<in>_<op>_<out>` into `(particles_in_ids, particles_out_ids, op)`
/// matching `do_parse_reaction_handle` in `rates/private/reaclib_support.f90`.
pub fn parse_reaction_handle(
    handle: &str,
) -> Result<(Vec<usize>, Vec<usize>, String), NetError> {
    let h = handle.trim();
    if !h.starts_with("r_") {
        return Err(NetError::InvalidReactionOrIso(h.to_string()));
    }
    let nuclides = NuclideCatalog::builtin_ref();
    let ih1 = nuclides.get_nuclide_index("h1").unwrap_or(2);
    let ihe4 = nuclides.get_nuclide_index("he4").unwrap_or(7);
    let ineut = nuclides.get_nuclide_index("neut").unwrap_or(1);

    let mut iso_ids = Vec::new();
    let mut num_in = 0_usize;
    let mut op = String::new();
    let mut doing_inputs = true;

    for part in h[2..].split('_') {
        if let Some(cid) = nuclides.get_nuclide_index(part) {
            iso_ids.push(cid);
        } else if doing_inputs {
            op = part.to_string();
            let mut extra_out: isize = -1;
            if op.len() == 2 {
                let b = op.as_bytes();
                let extra_in = match b[0] {
                    b'p' => ih1 as isize,
                    b'a' => ihe4 as isize,
                    b'n' => ineut as isize,
                    b'g' => 0,
                    _ => -1,
                };
                if extra_in >= 0 {
                    if extra_in > 0 {
                        if iso_ids.len() != 1 {
                            return Err(NetError::InvalidReactionOrIso(
                                h.to_string(),
                            ));
                        }
                        let ex = extra_in as usize;
                        let z0 =
                            nuclides.get_by_id(iso_ids[0]).map_or(0, |e| e.z);
                        let ze = nuclides.get_by_id(ex).map_or(0, |e| e.z);
                        if z0 >= ze {
                            let first = iso_ids[0];
                            iso_ids[0] = ex;
                            iso_ids.push(first);
                        } else {
                            iso_ids.push(ex);
                        }
                    }
                    extra_out = match b[1] {
                        b'p' => ih1 as isize,
                        b'a' => ihe4 as isize,
                        b'n' => ineut as isize,
                        b'g' => 0,
                        _ => {
                            return Err(NetError::InvalidReactionOrIso(
                                h.to_string(),
                            ))
                        }
                    };
                }
            } else if op == "wk-h1" {
                extra_out = ih1 as isize;
            } else if op == "wk-he4" {
                extra_out = ihe4 as isize;
            }
            num_in = iso_ids.len();
            doing_inputs = false;
            if extra_out > 0 {
                iso_ids.push(extra_out as usize);
            }
        } else {
            return Err(NetError::InvalidReactionOrIso(h.to_string()));
        }
    }
    if doing_inputs || num_in == 0 || iso_ids.len() <= num_in {
        return Err(NetError::InvalidReactionOrIso(h.to_string()));
    }
    Ok((iso_ids[..num_in].to_vec(), iso_ids[num_in..].to_vec(), op))
}

fn is_pp_reaction(h: &str) -> bool {
    matches!(
        h,
        "r_h1_h1_wk_h2"
            | "r_h1_h1_ec_h2"
            | "r_h2_pg_he3"
            | "r_he3_he3_to_h1_h1_he4"
            | "r_he3_ag_be7"
            | "r_be7_wk_li7"
            | "r_h1_li7_to_he4_he4"
            | "r_li7_pa_he4"
            | "r_be7_pg_b8"
            | "r_b8_wk_he4_he4"
    )
}

fn is_cno_reaction(h: &str) -> bool {
    matches!(
        h,
        "r_c12_pg_n13"
            | "r_c13_pg_n14"
            | "r_n13_wk_c13"
            | "r_n13_pg_o14"
            | "r_n14_pg_o15"
            | "r_n15_pg_o16"
            | "r_n15_pa_c12"
            | "r_o14_wk_n14"
            | "r_o14_ap_f17"
            | "r_o15_wk_n15"
            | "r_o16_pg_f17"
            | "r_o17_pg_f18"
            | "r_o17_pa_n14"
            | "r_o18_pg_f19"
            | "r_o18_pa_n15"
            | "r_f17_wk_o17"
            | "r_f17_pg_ne18"
            | "r_f18_pa_o15"
            | "r_f18_wk_o18"
            | "r_f19_pa_o16"
            | "r_ne18_wk_f18"
    )
}

fn pack_species(ids: &[usize]) -> (usize, [u8; 3], [u16; 3]) {
    let mut num = 0_usize;
    let mut counts = [0_u8; 3];
    let mut isos = [0_u16; 3];
    for &cid in ids {
        if num > 0 && usize::from(isos[num - 1]) == cid {
            counts[num - 1] += 1;
        } else if num < 3 {
            isos[num] = cid as u16;
            counts[num] = 1;
            num += 1;
        }
    }
    (num, counts, isos)
}

/// Synthesizes a [`ReactionInfo`] dynamically from `RatesCatalog` REACLIB handles
/// matching `do_add_reaction_from_reaclib` and `set_reaction_info` in
/// `rates/private/rates_initialize.f90`.
pub fn synthesize_reaction_info(
    handle: &str,
    id: usize,
) -> Result<ReactionInfo, NetError> {
    let cat = RatesCatalog::builtin();
    if let Some(ir) = cat.reaction_id_by_name(handle) {
        if let Ok(info) = cat.reaction_by_id(ir) {
            return Ok(info.clone());
        }
    }
    let (is_rev, terms) = cat
        .reaclib_terms(handle)
        .map_err(|_| NetError::InvalidReactionOrIso(handle.to_string()))?;
    let weak = terms.first().is_some_and(|t| t.rflag == b'w');
    let (p_in, p_out, _op) = parse_reaction_handle(handle)?;
    let particles_in = p_in.len();
    let particles_out = p_out.len();
    let (num_in, in_num, in_iso) = pack_species(&p_in);
    let (num_out, out_num, out_iso) = pack_species(&p_out);

    let nuclides = NuclideCatalog::builtin_ref();
    let mut std_q = 0.0_f64;
    for k in 0..num_in {
        if let Some(e) = nuclides.get_by_id(usize::from(in_iso[k])) {
            std_q += f64::from(in_num[k]) * e.mass_excess;
        }
    }
    for k in 0..num_out {
        if let Some(e) = nuclides.get_by_id(usize::from(out_iso[k])) {
            std_q -= f64::from(out_num[k]) * e.mass_excess;
        }
    }

    let iso_in = in_iso[num_in - 1];
    let cin1 = in_num[num_in - 1];
    let iso_out = out_iso[num_out - 1];
    let mut std_qneu = 0.0_f64;
    let mut weak_isos = [0_u16; 2];
    let mut weak_lowt_rate = -1.0e99_f64;

    if weak {
        weak_isos = [iso_in, iso_out];
        let weak_info = cat.weak_info_by_ids(iso_in, iso_out);
        if let Some((_, qneu)) = weak_info {
            std_qneu = qneu;
        }
        if let Ok((lam, _, _, _)) = eval_reaclib_handle(handle, 0.01, false) {
            weak_lowt_rate = lam;
        } else if let Some((hl, _)) = weak_info {
            if hl > 0.0 {
                weak_lowt_rate = LN2 / hl;
            }
        }
    }

    let (screening_isos, rho_exp) = if particles_in > 1 {
        let sc1 = in_iso[0];
        let sc2 = if cin1 > 1 { sc1 } else { in_iso[1] };
        ([sc1, sc2, 0], (particles_in - 1) as i8)
    } else {
        ([0, 0, 0], 0_i8)
    };

    let z_in = nuclides.get_by_id(usize::from(iso_in)).map_or(0, |e| e.z);
    let cat_name = if is_pp_reaction(handle) {
        "pp"
    } else if is_cno_reaction(handle) {
        "cno"
    } else if num_in == 1 && cin1 == 2 {
        match z_in {
            6 => "c12_c12",
            8 => "o16_o16",
            _ => "other",
        }
    } else if num_in > 1 {
        match z_in {
            6 => "c_alpha",
            7 => "n_alpha",
            8 => "o_alpha",
            10 => "ne_alpha",
            11 => "na_alpha",
            12 => "mg_alpha",
            14 => "si_alpha",
            16 => "s_alpha",
            18 => "ar_alpha",
            20 => "ca_alpha",
            22 => "ti_alpha",
            24 => "cr_alpha",
            26 => "fe_co_ni",
            _ => "other",
        }
    } else if particles_in == 1 && particles_out == 2 && !weak {
        "photo"
    } else {
        "other"
    };
    let cat_id = category_id(cat_name);
    let category = (if cat_id > 0 { cat_id } else { 24 }) as u8;

    Ok(ReactionInfo {
        id,
        name: handle.to_string(),
        op_code: 0,
        is_reverse: is_rev,
        target_handle: handle.to_string(),
        num_in,
        in_num,
        in_iso,
        num_out,
        out_num,
        out_iso,
        ye_exp: 0,
        rho_exp,
        category,
        screening_isos,
        weak_isos,
        std_q,
        std_qneu,
        weak_lowt_rate,
    })
}
