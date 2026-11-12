// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stellar model ASCII `.mod` reader/writer (`read_model.f90`, `write_model.f90`)
//! and binary `.photo` restart checkpoint serializer (`photo_in.f90`,
//! `photo_out.f90`).

use std::fmt::Write as _;

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{M_SUN, SECYER};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;
use crate::zams::parse_fortran_f64;

/// Binary `.photo` schema version matching `star_def_version = 19` in
/// `star_data/public/star_data_def.f90`.
pub const STAR_DEF_VERSION: i32 = 19;

/// Serializes a [`StarModel`] into an upstream MESA ASCII `.mod` string
/// (`write_model.f90`), preserving the exact `file_type` bitmask (`32812` for
/// `RSP`), core boundary headers, and `RSP` / `RSP2` zone columns.
///
/// # Errors
/// Returns [`StarError::IoError`] if formatting fails.
pub fn write_mod_string(model: &StarModel) -> Result<String, StarError> {
    let nz = model.nz;
    let write_v = model.controls.v_flag && model.v.len() == nz;
    let write_u = model.controls.u_flag && model.u.len() == nz;
    let write_rot = model.controls.rotation_flag
        && model.omega.len() == nz
        && model.j_rot.len() == nz;
    let write_rti = model.controls.rti_flag && model.aux.alpha_rti.len() == nz;
    let write_rsp = model.controls.rsp_flag
        && model.aux.rsp.et_rsp.len() == nz
        && model.aux.rsp.erad_rsp.len() == nz
        && model.aux.rsp.fr_rsp.len() == nz;
    let write_rsp2 = !write_rsp
        && model.controls.rsp2_flag
        && model.aux.rsp.hp_face.len() == nz;
    let write_mlt_vc = model.mlt_vc.len() == nz
        && (write_rsp || model.mlt_vc.iter().any(|&x| x != 0.0));

    let mut file_type = 4_u32;
    if write_v {
        file_type |= 8;
    }
    if write_rot {
        file_type |= 16 | 4096;
    }
    if write_mlt_vc {
        file_type |= 32;
    }
    if write_rsp2 {
        file_type |= 64;
    }
    if write_rti {
        file_type |= 128;
    }
    if write_u {
        file_type |= 512;
    }
    if write_rsp {
        file_type |= 32768;
    }

    let mut out = String::with_capacity(nz * 260 + 1200);
    writeln!(out, "! note: initial lines of file can contain comments\n!")
        .unwrap();
    writeln!(out, "{file_type:14} -- model for mesa/star, cgs units.\n")
        .unwrap();

    let (m_msun, dt_yrs, z_init) = (
        model.mstar / M_SUN,
        model.dt / SECYER,
        model.controls.initial_z,
    );
    let (mn, age, sp) = (model.model_number, model.star_age, model.species);
    writeln!(out, "             M/Msun      {m_msun:26.16E}").unwrap();
    writeln!(out, "       model_number      {mn:26}").unwrap();
    writeln!(out, "           star_age      {age:26.16E}").unwrap();
    writeln!(out, "          initial_z      {z_init:26.16E}").unwrap();
    writeln!(out, "           n_shells      {nz:26}").unwrap();
    writeln!(out, "           net_name   '{}'", model.job.net_name).unwrap();
    writeln!(out, "            species      {sp:26}").unwrap();
    let c_err = model.aux.cumulative_energy_error;
    for (k_str, val, show) in [
        ("xmstar", model.xmstar, model.m_center > 0.0),
        (
            "M_center",
            model.m_center,
            model.m_center > 0.0 && !write_rsp,
        ),
        ("R_center", model.r_center, model.r_center > 0.0),
        ("L_center", model.l_center, model.l_center > 0.0),
        ("v_center", model.v_center, model.v_center != 0.0),
        (
            "tau_factor",
            model.controls.tau_factor,
            model.controls.tau_factor != 1.0,
        ),
        (
            "opacity_factor",
            model.controls.opacity_factor,
            model.controls.opacity_factor != 1.0,
        ),
        (
            "total_energy",
            model.aux.total_energy,
            write_rsp && model.aux.total_energy != 0.0,
        ),
        ("cumulative_energy_error", c_err, c_err != 0.0),
    ] {
        if show {
            writeln!(out, "{k_str:>19}      {val:26.16E}").unwrap();
        }
    }
    writeln!(out, "               Teff      {:26.16E}", model.teff).unwrap();
    if model.num_retries > 0 || write_rsp {
        writeln!(out, "        num_retries      {:26}", model.num_retries)
            .unwrap();
    }
    if !write_rsp {
        writeln!(out, "   year_for_time_step      {dt_yrs:26.16E}").unwrap();
    }
    writeln!(out).unwrap();

    write!(
        out,
        "        lnd                       lnT                       lnR"
    )
    .unwrap();
    if write_rsp {
        for col in ["et_RSP", "erad_RSP", "Fr_RSP"] {
            write!(out, "  {col:>24}").unwrap();
        }
    } else if write_rsp2 {
        for col in ["w", "Hp"] {
            write!(out, "  {col:>24}").unwrap();
        }
    }
    for col in ["L", "dq"] {
        write!(out, "  {col:>24}").unwrap();
    }
    for (col_name, show) in [
        ("v", write_v),
        ("u", write_u),
        ("omega", write_rot),
        ("j_rot", write_rot),
        ("alpha_RTI", write_rti),
        ("mlt_vc", write_mlt_vc),
    ] {
        if show {
            write!(out, "  {col_name:>24}").unwrap();
        }
    }
    for name in &model.iso_names {
        write!(out, "  {name:>24}").unwrap();
    }
    writeln!(out).unwrap();

    for k in 0..nz {
        write!(
            out,
            "{:5}  {:24.16E}  {:24.16E}  {:24.16E}",
            k + 1,
            model.lnd[k],
            model.ln_t[k],
            model.ln_r[k],
        )
        .unwrap();
        if write_rsp {
            write!(
                out,
                "  {:24.16E}  {:24.16E}  {:24.16E}",
                model.aux.rsp.et_rsp[k],
                model.aux.rsp.erad_rsp[k],
                model.aux.rsp.fr_rsp[k],
            )
            .unwrap();
        } else if write_rsp2 {
            let w_k = model
                .aux
                .etrb
                .get(k)
                .copied()
                .unwrap_or(0.0)
                .max(0.0)
                .sqrt();
            let hp_k = if model.aux.rsp.hp_face[k] > 0.0 {
                model.aux.rsp.hp_face[k]
            } else {
                model.scale_height.get(k).copied().unwrap_or(0.0)
            };
            write!(out, "  {w_k:24.16E}  {hp_k:24.16E}").unwrap();
        }
        write!(out, "  {:24.16E}  {:24.16E}", model.lum[k], model.dq[k])
            .unwrap();
        for (val, show) in [
            (model.v.get(k).copied().unwrap_or(0.0), write_v),
            (model.u.get(k).copied().unwrap_or(0.0), write_u),
            (model.omega.get(k).copied().unwrap_or(0.0), write_rot),
            (model.j_rot.get(k).copied().unwrap_or(0.0), write_rot),
            (
                model.aux.alpha_rti.get(k).copied().unwrap_or(0.0),
                write_rti,
            ),
            (model.mlt_vc.get(k).copied().unwrap_or(0.0), write_mlt_vc),
        ] {
            if show {
                write!(out, "  {val:24.16E}").unwrap();
            }
        }
        for &x_j in model.cell_xa(k) {
            write!(out, "  {x_j:24.16E}").unwrap();
        }
        writeln!(out).unwrap();
    }

    let (m_old, dt_old, dt_next) =
        (model.mstar_old, model.dt_old, model.dt_next);
    writeln!(out, "\n        previous model").unwrap();
    writeln!(out, "   previous n_shells      {nz:26}").unwrap();
    writeln!(out, "  previous mass (grams)   {m_old:26.16E}").unwrap();
    writeln!(out, "   timestep (seconds)     {dt_old:26.16E}").unwrap();
    writeln!(out, "    dt_next (seconds)     {dt_next:26.16E}").unwrap();
    Ok(out)
}

/// Parses an upstream MESA ASCII `.mod` string into a [`StarModel`]
/// (`read_model.f90`), supporting all standard, `RSP` (`32812`), and `RSP2`
/// `.mod` column layouts.
///
/// # Errors
/// Returns [`StarError::IoError`] if the `.mod` text is malformed.
pub fn read_mod_str(
    text: &str,
    mut controls: StarControls,
    mut job: StarJobControls,
) -> Result<StarModel, StarError> {
    let mut lines = text.lines();
    let (mut m_msun, mut star_age, mut teff) =
        (controls.initial_mass, 0.0_f64, 5777.0_f64);
    let (mut model_number, mut nz, mut num_retries) =
        (1_usize, 0_usize, 0_usize);
    let (mut header_species, mut dt_yrs) = (None, None);
    let (mut xmstar_opt, mut m_center_opt) = (None, None);
    let (mut r_center, mut l_center, mut v_center, mut cum_energy_err) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut total_energy = 0.0_f64;
    let mut header_cols: Vec<String> = Vec::new();

    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('!') {
            continue;
        }
        if trimmed.starts_with("lnd ") {
            header_cols = trimmed
                .split_whitespace()
                .map(ToString::to_string)
                .collect();
            break;
        }
        let Some((key, rest)) = trimmed.split_once(char::is_whitespace) else {
            continue;
        };
        let Some(v) = rest.split_whitespace().next() else {
            continue;
        };
        match key {
            "M/Msun" => {
                m_msun = parse_fortran_f64(v)?;
                controls.initial_mass = m_msun;
            }
            "model_number" => model_number = v.parse::<usize>().unwrap_or(1),
            "star_age" => star_age = parse_fortran_f64(v)?,
            "initial_z" | "initial_Z" => {
                controls.initial_z = parse_fortran_f64(v)?;
            }
            "initial_y" | "initial_Y" => {
                controls.initial_y = parse_fortran_f64(v)?;
            }
            "n_shells" => nz = v.parse().unwrap_or(0),
            "net_name" => {
                job.net_name = v.trim_matches(['\'', '"']).to_string();
            }
            "species" => header_species = v.parse::<usize>().ok(),
            "Teff" => teff = parse_fortran_f64(v)?,
            "year_for_time_step" => dt_yrs = Some(parse_fortran_f64(v)?),
            "xmstar" => xmstar_opt = Some(parse_fortran_f64(v)?),
            "M_center" => m_center_opt = Some(parse_fortran_f64(v)?),
            "R_center" => r_center = parse_fortran_f64(v)?,
            "L_center" => l_center = parse_fortran_f64(v)?,
            "v_center" => v_center = parse_fortran_f64(v)?,
            "total_energy" => total_energy = parse_fortran_f64(v)?,
            "cumulative_energy_error" => cum_energy_err = parse_fortran_f64(v)?,
            "tau_factor" if !job.set_tau_factor => {
                controls.tau_factor = parse_fortran_f64(v)?;
            }
            "opacity_factor" => controls.opacity_factor = parse_fortran_f64(v)?,
            "num_retries" => num_retries = v.parse::<usize>().unwrap_or(0),
            _ => {}
        }
    }

    if nz == 0 || header_cols.len() < 5 {
        return Err(StarError::IoError(String::from("malformed .mod header")));
    }
    let find_col = |name: &str| -> Result<usize, StarError> {
        header_cols
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| StarError::IoError(format!("missing {name} col")))
    };
    let (lnd_idx, lnt_idx) = (find_col("lnd")?, find_col("lnT")?);
    let (lnr_idx, lum_idx, dq_idx) =
        (find_col("lnR")?, find_col("L")?, find_col("dq")?);

    let catalog = NuclideCatalog::builtin_ref();
    let iso_start_idx = if let Some(sp) =
        header_species.filter(|&s| s > 0 && header_cols.len() > s)
    {
        header_cols.len() - sp
    } else {
        (dq_idx + 1..header_cols.len())
            .find(|&i| catalog.get_nuclide_index(&header_cols[i]).is_some())
            .unwrap_or(dq_idx + 1)
    };
    let pre_iso_cols = &header_cols[..iso_start_idx];
    let col_pos = |name: &str| pre_iso_cols.iter().position(|c| c == name);
    let (v_idx, u_idx) = (col_pos("v"), col_pos("u"));
    let (omega_idx, j_rot_idx) = (col_pos("omega"), col_pos("j_rot"));
    let (w_div_wc_idx, w_idx, hp_idx) =
        (col_pos("w_div_wc"), col_pos("w"), col_pos("Hp"));
    let (et_rsp_idx, erad_rsp_idx, fr_rsp_idx) =
        (col_pos("et_RSP"), col_pos("erad_RSP"), col_pos("Fr_RSP"));
    let rti_idx = pre_iso_cols
        .iter()
        .position(|c| c.eq_ignore_ascii_case("alpha_rti"));
    let vc_idx = col_pos("conv_vel").or_else(|| col_pos("mlt_vc"));

    controls.v_flag |= v_idx.is_some() && !job.change_v_flag;
    controls.u_flag |= u_idx.is_some() && !job.change_u_flag;
    controls.rotation_flag |= omega_idx.is_some() && !job.change_rotation_flag;
    controls.rti_flag |= rti_idx.is_some() && !job.change_initial_rti_flag;
    controls.rsp_flag = if controls.rsp2.change_rsp_flag {
        controls.rsp2.new_rsp_flag
    } else {
        controls.rsp_flag || et_rsp_idx.is_some()
    };
    controls.rsp2_flag = if controls.rsp2.change_rsp2_flag {
        controls.rsp2.new_rsp2_flag
    } else {
        controls.rsp2_flag || (w_idx.is_some() && hp_idx.is_some())
    };
    if controls.rsp2_flag {
        controls.rsp_flag = false;
    }

    let iso_names = &header_cols[iso_start_idx..];
    let species = iso_names.len();
    let chem_id: Vec<usize> = iso_names
        .iter()
        .map(|n| catalog.get_nuclide_index(n).unwrap_or(1))
        .collect();

    let mut model = StarModel::new(nz, chem_id, job.clone(), controls);
    model.mstar = m_msun * M_SUN;
    (model.mstar_start, model.mstar_old) = (model.mstar, model.mstar);
    if let Some(mc) = m_center_opt.filter(|&m| m > 0.0) {
        model.m_center = mc;
        model.xmstar = (model.mstar - mc).max(0.0);
    }
    if let Some(xm) = xmstar_opt.filter(|&x| x > 0.0) {
        model.xmstar = xm;
        model.m_center = (model.mstar - xm).max(0.0);
    }
    (model.r_center, model.l_center, model.v_center) =
        (r_center, l_center, v_center);
    model.aux.total_energy = total_energy;
    model.aux.cumulative_energy_error = cum_energy_err;
    (model.model_number, model.num_retries) = (model_number, num_retries);
    (model.star_age, model.teff) = (star_age, teff);
    let init_dt = dt_yrs.unwrap_or(1000.0) * SECYER;
    (model.dt, model.dt_next, model.dt_old) = (init_dt, init_dt, init_dt);

    let total_cols = header_cols.len();
    let mut k = 0_usize;
    while k < nz {
        let Some(line) = lines.next() else {
            return Err(StarError::IoError(format!(
                "unexpected EOF at zone {k}/{nz}"
            )));
        };
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        if toks.len() < 1 + total_cols {
            return Err(StarError::IoError(format!(
                "short zone row in .mod at zone {}: expected {}, got {}",
                k + 1,
                1 + total_cols,
                toks.len()
            )));
        }
        let offset = 1;
        for (opt_idx, dst) in [
            (Some(lnd_idx), &mut model.lnd[k]),
            (Some(lnt_idx), &mut model.ln_t[k]),
            (Some(lnr_idx), &mut model.ln_r[k]),
            (Some(lum_idx), &mut model.lum[k]),
            (Some(dq_idx), &mut model.dq[k]),
            (v_idx, &mut model.v[k]),
            (u_idx, &mut model.u[k]),
            (omega_idx, &mut model.omega[k]),
            (j_rot_idx, &mut model.j_rot[k]),
            (w_div_wc_idx, &mut model.aux.w_div_wc[k]),
            (rti_idx, &mut model.aux.alpha_rti[k]),
            (erad_rsp_idx, &mut model.aux.rsp.erad_rsp[k]),
            (fr_rsp_idx, &mut model.aux.rsp.fr_rsp[k]),
        ] {
            if let Some(idx) = opt_idx {
                *dst = parse_fortran_f64(toks[offset + idx])?;
            }
        }
        if let Some(idx) = w_idx {
            let w_val = parse_fortran_f64(toks[offset + idx])?;
            let et = w_val * w_val;
            model.aux.etrb[k] = et;
            model.aux.rsp.et_rsp[k] = et;
        } else if let Some(idx) = et_rsp_idx {
            let et = parse_fortran_f64(toks[offset + idx])?;
            model.aux.etrb[k] = et;
            model.aux.rsp.et_rsp[k] = et;
        }
        if let Some(idx) = hp_idx {
            let hp = parse_fortran_f64(toks[offset + idx])?;
            model.aux.rsp.hp_face[k] = hp;
            model.scale_height[k] = hp;
        }
        if let Some(idx) = vc_idx {
            let vc = parse_fortran_f64(toks[offset + idx])?;
            (model.mlt_vc[k], model.conv_vel[k]) = (vc, vc);
        }
        for j in 0..species {
            model.xa[k * species + j] =
                parse_fortran_f64(toks[offset + iso_start_idx + j])?;
        }
        k += 1;
    }

    let (mut parsed_mstar_old, mut parsed_dt_old) =
        (model.mstar_old, model.dt_old);
    for line in lines {
        let t = line.trim();
        let Some(v) = t.split_whitespace().last() else {
            continue;
        };
        if t.starts_with("previous mass (grams)") {
            parsed_mstar_old = parse_fortran_f64(v)?;
        } else if t.starts_with("timestep (seconds)") {
            parsed_dt_old = parse_fortran_f64(v)?;
            if dt_yrs.is_none() {
                model.dt = parsed_dt_old;
            }
        } else if t.starts_with("dt_next (seconds)") {
            model.dt_next = parse_fortran_f64(v)?;
            if dt_yrs.is_none() {
                model.dt = model.dt_next;
            }
        }
    }

    let target_net = if job.change_net && !job.new_net_name.is_empty() {
        job.new_net_name.clone()
    } else {
        model.job.net_name.clone()
    };
    if let Ok(net) = crate::micro::get_or_load_net(&target_net, None) {
        if net.chem_id != model.chem_id {
            crate::micro::change_model_network(&mut model, &target_net, None)?;
        }
    }

    let saved_dq = model.dq.clone();
    let saved_vc = vc_idx.map(|_| model.mlt_vc.clone());
    let saved_hp = hp_idx.map(|_| model.aux.rsp.hp_face.clone());
    if model.controls.rsp_flag {
        if (saved_dq.iter().sum::<f64>() - 1.0).abs() <= 1.0e-12 {
            model.dq.clone_from(&saved_dq);
            model.dq_old = saved_dq;
        }
        model.set_qs();
        model.set_xqs();
        model.set_m_and_dm();
        model.set_m_grav_and_grav();
        model.pack_xh();
        crate::micro::set_basic_vars(&mut model)?;
        model.save_start_values();
        model.save_old_values();
    } else {
        model.finish_init_model()?;
        if (saved_dq.iter().sum::<f64>() - 1.0).abs() <= 1.0e-12 {
            model.dq.clone_from(&saved_dq);
            model.dq_old = saved_dq;
            model.set_qs();
            model.set_xqs();
            model.set_m_and_dm();
            model.set_m_grav_and_grav();
        }
    }
    if let Some(vc) = saved_vc {
        model.conv_vel.clone_from(&vc);
        model.aux.mlt_vc_old.clone_from(&vc);
        model.mlt_vc = vc;
    }
    if let Some(hp) = saved_hp {
        model.scale_height.clone_from(&hp);
        model.aux.rsp.hp_face = hp;
    }
    (model.mstar_old, model.dt_old) = (parsed_mstar_old, parsed_dt_old);

    Ok(model)
}

pub use crate::photo_io::{
    compute_photo_checksum, read_photo_bytes, write_photo_bytes,
};
