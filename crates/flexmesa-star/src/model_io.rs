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
/// (`write_model.f90`), preserving optional core boundary headers (`xmstar`,
/// `R_center`, `L_center`, `v_center`) and active dynamic/rotational columns
/// (`v`, `u`, `omega`, `j_rot`).
///
/// # Errors
/// Returns [`StarError::IoError`] if formatting fails.
pub fn write_mod_string(model: &StarModel) -> Result<String, StarError> {
    let mut out = String::with_capacity(model.nz * 200 + 1024);
    writeln!(out, "! note: initial lines of file can contain comments")
        .unwrap();
    writeln!(out, "!").unwrap();
    writeln!(out, "           1 -- model with uniform dq").unwrap();
    writeln!(out).unwrap();

    let m_msun = model.mstar / M_SUN;
    let dt_yrs = model.dt / SECYER;
    writeln!(out, "             M/Msun      {m_msun:26.16E}").unwrap();
    writeln!(out, "       model_number      {:26}", model.model_number)
        .unwrap();
    writeln!(out, "           star_age      {:26.16E}", model.star_age)
        .unwrap();
    writeln!(
        out,
        "          initial_z      {:26.16E}",
        model.controls.initial_z
    )
    .unwrap();
    writeln!(out, "           n_shells      {:26}", model.nz).unwrap();
    writeln!(out, "           net_name   '{}'", model.job.net_name).unwrap();
    writeln!(out, "            species      {:26}", model.species).unwrap();
    for (k_str, val, show) in [
        ("xmstar", model.xmstar, model.m_center > 0.0),
        ("R_center", model.r_center, model.r_center > 0.0),
        ("L_center", model.l_center, model.l_center > 0.0),
        ("v_center", model.v_center, model.v_center != 0.0),
    ] {
        if show {
            writeln!(out, "{k_str:>19}      {val:26.16E}").unwrap();
        }
    }
    writeln!(out, "               Teff      {:26.16E}", model.teff).unwrap();
    writeln!(out, "   year_for_time_step      {dt_yrs:26.16E}").unwrap();
    writeln!(out).unwrap();

    let write_v = model.controls.v_flag && model.v.len() == model.nz;
    let write_u = model.controls.u_flag && model.u.len() == model.nz;
    let write_rot = model.controls.rotation_flag
        && model.omega.len() == model.nz
        && model.j_rot.len() == model.nz;

    write!(
        out,
        "        lnd                       lnT                       lnR                       L                         dq"
    )
    .unwrap();
    if write_v {
        write!(out, "  {:>24}", "v").unwrap();
    }
    if write_u {
        write!(out, "  {:>24}", "u").unwrap();
    }
    if write_rot {
        write!(out, "  {:>24}  {:>24}", "omega", "j_rot").unwrap();
    }
    for name in &model.iso_names {
        write!(out, "  {name:>24}").unwrap();
    }
    writeln!(out).unwrap();

    for k in 0..model.nz {
        write!(
            out,
            "{:5}  {:24.16E}  {:24.16E}  {:24.16E}  {:24.16E}  {:24.16E}",
            k + 1,
            model.lnd[k],
            model.ln_t[k],
            model.ln_r[k],
            model.lum[k],
            model.dq[k],
        )
        .unwrap();
        if write_v {
            write!(out, "  {:24.16E}", model.v[k]).unwrap();
        }
        if write_u {
            write!(out, "  {:24.16E}", model.u[k]).unwrap();
        }
        if write_rot {
            write!(
                out,
                "  {:24.16E}  {:24.16E}",
                model.omega[k], model.j_rot[k]
            )
            .unwrap();
        }
        for &x_j in model.cell_xa(k) {
            write!(out, "  {x_j:24.16E}").unwrap();
        }
        writeln!(out).unwrap();
    }

    writeln!(out).unwrap();
    writeln!(out, "        previous model").unwrap();
    writeln!(out, "   previous n_shells      {:26}", model.nz).unwrap();
    writeln!(out, "  previous mass (grams)   {:26.16E}", model.mstar_old)
        .unwrap();
    writeln!(out, "   timestep (seconds)     {:26.16E}", model.dt_old).unwrap();
    writeln!(out, "    dt_next (seconds)     {:26.16E}", model.dt_next)
        .unwrap();

    Ok(out)
}

/// Parses an upstream MESA ASCII `.mod` string into a [`StarModel`]
/// (`read_model.f90`), supporting all 16 upstream `.mod` column layouts
/// (`lnd`, `lnT`, `lnR`, `et_RSP`/`erad_RSP`/`Fr_RSP`, `w`/`Hp`, `L`, `dq`,
/// `v`, `omega`, `j_rot`, `D_omega`, `am_nu_rot`, `u`, `alpha_RTI`,
/// `conv_vel`/`mlt_vc`, and `species` nuclide columns).
///
/// # Errors
/// Returns [`StarError::IoError`] if the `.mod` text is malformed.
pub fn read_mod_str(
    text: &str,
    mut controls: StarControls,
    mut job: StarJobControls,
) -> Result<StarModel, StarError> {
    let mut lines = text.lines();
    let mut m_msun = controls.initial_mass;
    let mut model_number = 1_usize;
    let mut star_age = 0.0_f64;
    let mut nz = 0_usize;
    let mut header_species: Option<usize> = None;
    let mut teff = 5777.0_f64;
    let mut dt_yrs: Option<f64> = None;
    let mut xmstar_opt: Option<f64> = None;
    let mut r_center = 0.0_f64;
    let mut l_center = 0.0_f64;
    let mut v_center = 0.0_f64;
    let mut num_retries = 0_usize;
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
            "n_shells" => {
                nz = v
                    .parse()
                    .map_err(|e| StarError::IoError(format!("{e}")))?;
            }
            "net_name" => {
                job.net_name = v.trim_matches(['\'', '"']).to_string();
            }
            "species" => header_species = v.parse::<usize>().ok(),
            "Teff" => teff = parse_fortran_f64(v)?,
            "year_for_time_step" => dt_yrs = Some(parse_fortran_f64(v)?),
            "xmstar" => xmstar_opt = Some(parse_fortran_f64(v)?),
            "R_center" => r_center = parse_fortran_f64(v)?,
            "L_center" => l_center = parse_fortran_f64(v)?,
            "v_center" => v_center = parse_fortran_f64(v)?,
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
    let lnd_idx = find_col("lnd")?;
    let lnt_idx = find_col("lnT")?;
    let lnr_idx = find_col("lnR")?;
    let lum_idx = find_col("L")?;
    let dq_idx = find_col("dq")?;

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
    let v_idx = pre_iso_cols.iter().position(|c| c == "v");
    let u_idx = pre_iso_cols.iter().position(|c| c == "u");
    let omega_idx = pre_iso_cols.iter().position(|c| c == "omega");
    let j_rot_idx = pre_iso_cols.iter().position(|c| c == "j_rot");
    let vc_idx = pre_iso_cols
        .iter()
        .position(|c| c == "conv_vel" || c == "mlt_vc");

    if v_idx.is_some() && !job.change_v_flag {
        controls.v_flag = true;
    }
    if u_idx.is_some() && !job.change_u_flag {
        controls.u_flag = true;
    }
    if omega_idx.is_some() && !job.change_rotation_flag {
        controls.rotation_flag = true;
    }

    let iso_names = &header_cols[iso_start_idx..];
    let species = iso_names.len();
    let chem_id: Vec<usize> = iso_names
        .iter()
        .map(|n| catalog.get_nuclide_index(n).unwrap_or(1))
        .collect();

    let mut model = StarModel::new(nz, chem_id, job.clone(), controls);
    model.eos_config.use_free_eos = false;
    model.eos_config.use_opal_scvh = false;
    model.mstar = m_msun * M_SUN;
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    if let Some(xm) = xmstar_opt.filter(|&x| x > 0.0) {
        model.xmstar = xm;
        model.m_center = (model.mstar - xm).max(0.0);
    }
    model.r_center = r_center;
    model.l_center = l_center;
    model.v_center = v_center;
    model.model_number = model_number;
    model.star_age = star_age;
    model.teff = teff;
    model.num_retries = num_retries;
    let init_dt = dt_yrs.unwrap_or(1000.0) * SECYER;
    model.dt = init_dt;
    model.dt_next = init_dt;
    model.dt_old = init_dt;

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
        model.lnd[k] = parse_fortran_f64(toks[offset + lnd_idx])?;
        model.ln_t[k] = parse_fortran_f64(toks[offset + lnt_idx])?;
        model.ln_r[k] = parse_fortran_f64(toks[offset + lnr_idx])?;
        model.lum[k] = parse_fortran_f64(toks[offset + lum_idx])?;
        model.dq[k] = parse_fortran_f64(toks[offset + dq_idx])?;
        if let Some(idx) = v_idx {
            model.v[k] = parse_fortran_f64(toks[offset + idx])?;
        }
        if let Some(idx) = u_idx {
            model.u[k] = parse_fortran_f64(toks[offset + idx])?;
        }
        if let Some(idx) = omega_idx {
            model.omega[k] = parse_fortran_f64(toks[offset + idx])?;
        }
        if let Some(idx) = j_rot_idx {
            model.j_rot[k] = parse_fortran_f64(toks[offset + idx])?;
        }
        if let Some(idx) = vc_idx {
            let vc = parse_fortran_f64(toks[offset + idx])?;
            model.mlt_vc[k] = vc;
            model.conv_vel[k] = vc;
        }
        for j in 0..species {
            model.xa[k * species + j] =
                parse_fortran_f64(toks[offset + iso_start_idx + j])?;
        }
        k += 1;
    }

    let mut parsed_mstar_old = model.mstar_old;
    let mut parsed_dt_old = model.dt_old;
    for line in lines {
        let t = line.trim();
        if t.starts_with("previous mass (grams)") {
            if let Some(v) = t.split_whitespace().last() {
                parsed_mstar_old = parse_fortran_f64(v)?;
            }
        } else if t.starts_with("timestep (seconds)") {
            if let Some(v) = t.split_whitespace().last() {
                let dt_s = parse_fortran_f64(v)?;
                parsed_dt_old = dt_s;
                if dt_yrs.is_none() {
                    model.dt = dt_s;
                }
            }
        } else if t.starts_with("dt_next (seconds)") {
            if let Some(v) = t.split_whitespace().last() {
                model.dt_next = parse_fortran_f64(v)?;
                if dt_yrs.is_none() {
                    model.dt = model.dt_next;
                }
            }
        }
    }

    // Align `model.chem_id` and `model.xa` ordering with `NetGeneralInfo.chem_id`
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

    model.finish_init_model()?;
    model.mstar_old = parsed_mstar_old;
    model.dt_old = parsed_dt_old;

    Ok(model)
}

/// Computes a deterministic 64-bit FNV-1a checksum of `.photo` restart bytes
/// (`./ck` in upstream MESA `test_suite_helpers`).
#[must_use]
pub fn compute_photo_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3_u64);
    }
    hash
}

/// Serializes a [`StarModel`] into binary `.photo` restart bytes with
/// `STAR_DEF_VERSION = 19` and `part_number = 1..7` sentinels (`photo_out.f90`).
#[must_use]
pub fn write_photo_bytes(model: &StarModel) -> Vec<u8> {
    let mut buf =
        Vec::with_capacity(model.nz * (8 * (12 + model.species)) + 256);
    let write_i32 =
        |buf: &mut Vec<u8>, v: i32| buf.extend_from_slice(&v.to_le_bytes());
    let write_u64 =
        |buf: &mut Vec<u8>, v: u64| buf.extend_from_slice(&v.to_le_bytes());
    let write_f64_slice = |buf: &mut Vec<u8>, s: &[f64]| {
        for &v in s {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    };

    write_i32(&mut buf, STAR_DEF_VERSION);

    // Part 1: grid & species dimensions and chem_id
    write_i32(&mut buf, 1);
    write_u64(&mut buf, model.nz as u64);
    write_u64(&mut buf, model.species as u64);
    write_u64(&mut buf, model.layout.nvar_hydro as u64);
    for &cid in &model.chem_id {
        write_u64(&mut buf, cid as u64);
    }

    // Part 2: primary hydro and rotational arrays (`lnd`, `ln_t`, `ln_r`, `lum`, `v`, `u`, `dq`, `xa`, `omega`, `j_rot`)
    write_i32(&mut buf, 2);
    for s in [&model.lnd, &model.ln_t, &model.ln_r, &model.lum] {
        write_f64_slice(&mut buf, s);
    }
    for s in [&model.v, &model.u, &model.dq, &model.xa] {
        write_f64_slice(&mut buf, s);
    }
    write_u64(&mut buf, model.omega.len() as u64);
    write_f64_slice(&mut buf, &model.omega);
    write_u64(&mut buf, model.j_rot.len() as u64);
    write_f64_slice(&mut buf, &model.j_rot);

    // Part 3: old-step arrays
    write_i32(&mut buf, 3);
    write_u64(&mut buf, model.lnd_old.len() as u64);
    for s in [&model.lnd_old, &model.ln_t_old, &model.ln_r_old] {
        write_f64_slice(&mut buf, s);
    }
    for s in [&model.lum_old, &model.dq_old, &model.xa_old] {
        write_f64_slice(&mut buf, s);
    }

    // Part 4: evolutionary scalar state (including H211 filter memory `dt_limit_ratio_old`)
    write_i32(&mut buf, 4);
    write_u64(&mut buf, model.model_number as u64);
    for v in [model.star_age, model.star_age_old, model.dt] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [model.dt_next, model.dt_old, model.dt_limit_ratio_old] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [model.mstar, model.mstar_old, model.mstar_dot] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [model.m_center, model.r_center, model.l_center] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [model.v_center, model.teff, model.tau_base] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    write_u64(&mut buf, model.total_num_solver_iterations as u64);
    write_u64(&mut buf, model.num_retries as u64);

    // Part 5..7 sentinels (`photo_out.f90`)
    write_i32(&mut buf, 5);
    write_i32(&mut buf, 6);
    write_i32(&mut buf, 7);

    buf
}

/// Restores a [`StarModel`] from binary `.photo` restart bytes with bit-identical
/// floating-point state (`photo_in.f90`).
///
/// # Errors
/// Returns [`StarError::IoError`] if version or sentinel checks fail.
pub fn read_photo_bytes(
    bytes: &[u8],
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    let mut pos = 0_usize;
    let eof = |msg: &str| StarError::IoError(msg.to_string());
    let read_i32 = |pos: &mut usize| -> Result<i32, StarError> {
        let s = bytes.get(*pos..*pos + 4).ok_or_else(|| eof("EOF i32"))?;
        *pos += 4;
        Ok(i32::from_le_bytes(s.try_into().unwrap()))
    };
    let read_u64 = |pos: &mut usize| -> Result<u64, StarError> {
        let s = bytes.get(*pos..*pos + 8).ok_or_else(|| eof("EOF u64"))?;
        *pos += 8;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    };
    let read_f64_vec = |pos: &mut usize,
                        n: usize|
     -> Result<Vec<f64>, StarError> {
        let end = *pos + 8 * n;
        let slice = bytes.get(*pos..end).ok_or_else(|| eof("EOF f64 slice"))?;
        *pos = end;
        Ok(slice
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c))
            .collect())
    };
    let expect_part =
        |pos: &mut usize, expected: i32| -> Result<(), StarError> {
            let p = read_i32(pos)?;
            if p != expected {
                return Err(StarError::IoError(format!(
                "photo sentinel mismatch: expected part {expected}, got {p}"
            )));
            }
            Ok(())
        };

    let version = read_i32(&mut pos)?;
    if version != STAR_DEF_VERSION {
        return Err(StarError::IoError(format!(
            "unsupported photo version {version} (expected {STAR_DEF_VERSION})"
        )));
    }

    expect_part(&mut pos, 1)?;
    let nz = read_u64(&mut pos)? as usize;
    let species = read_u64(&mut pos)? as usize;
    let _nvar_hydro = read_u64(&mut pos)? as usize;
    let mut chem_id = Vec::with_capacity(species);
    for _ in 0..species {
        chem_id.push(read_u64(&mut pos)? as usize);
    }

    let mut model = StarModel::new(nz, chem_id, job, controls);
    model.eos_config.use_free_eos = false;
    model.eos_config.use_opal_scvh = false;

    expect_part(&mut pos, 2)?;
    model.lnd = read_f64_vec(&mut pos, nz)?;
    model.ln_t = read_f64_vec(&mut pos, nz)?;
    model.ln_r = read_f64_vec(&mut pos, nz)?;
    model.lum = read_f64_vec(&mut pos, nz)?;
    model.v = read_f64_vec(&mut pos, nz)?;
    model.u = read_f64_vec(&mut pos, nz)?;
    model.dq = read_f64_vec(&mut pos, nz)?;
    model.xa = read_f64_vec(&mut pos, nz * species)?;
    let n_omega = read_u64(&mut pos)? as usize;
    model.omega = read_f64_vec(&mut pos, n_omega)?;
    let n_jrot = read_u64(&mut pos)? as usize;
    model.j_rot = read_f64_vec(&mut pos, n_jrot)?;

    expect_part(&mut pos, 3)?;
    let nz_old = read_u64(&mut pos)? as usize;
    model.lnd_old = read_f64_vec(&mut pos, nz_old)?;
    model.ln_t_old = read_f64_vec(&mut pos, nz_old)?;
    model.ln_r_old = read_f64_vec(&mut pos, nz_old)?;
    model.lum_old = read_f64_vec(&mut pos, nz_old)?;
    model.dq_old = read_f64_vec(&mut pos, nz_old)?;
    model.xa_old = read_f64_vec(&mut pos, nz_old * species)?;

    expect_part(&mut pos, 4)?;
    model.model_number = read_u64(&mut pos)? as usize;
    let s = read_f64_vec(&mut pos, 15)?;
    (model.star_age, model.star_age_old, model.dt) = (s[0], s[1], s[2]);
    (model.dt_next, model.dt_old, model.dt_limit_ratio_old) =
        (s[3], s[4], s[5]);
    (model.mstar, model.mstar_old, model.mstar_dot) = (s[6], s[7], s[8]);
    (model.m_center, model.r_center, model.l_center) = (s[9], s[10], s[11]);
    (model.v_center, model.teff, model.tau_base) = (s[12], s[13], s[14]);
    model.total_num_solver_iterations = read_u64(&mut pos)? as usize;
    model.num_retries = read_u64(&mut pos)? as usize;

    expect_part(&mut pos, 5)?;
    expect_part(&mut pos, 6)?;
    expect_part(&mut pos, 7)?;

    model.xmstar = (model.mstar - model.m_center).max(0.0);
    model.mstar_start = model.mstar;
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.set_m_grav_and_grav();
    model.pack_xh();
    crate::hydro_vars::set_hydro_vars(&mut model, false, false)?;
    let dt_photo = model.dt;
    crate::struct_burn_mix::refresh_mixing_state(&mut model, dt_photo, &[]);
    model.save_start_values();

    Ok(model)
}
