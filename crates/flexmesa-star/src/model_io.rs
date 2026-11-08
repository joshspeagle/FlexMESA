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
/// (`write_model.f90`).
///
/// # Errors
/// Returns [`StarError::IoError`] if formatting fails.
pub fn write_mod_string(model: &StarModel) -> Result<String, StarError> {
    let mut out = String::with_capacity(model.nz * 180 + 1024);
    writeln!(out, "! note: initial lines of file can contain comments")
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "!").map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "           1 -- model with uniform dq")
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out).map_err(|e| StarError::IoError(e.to_string()))?;

    let m_msun = model.mstar / M_SUN;
    let dt_yrs = model.dt / SECYER;
    writeln!(out, "             M/Msun      {m_msun:26.16E}")
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "       model_number      {:26}", model.model_number)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "           star_age      {:26.16E}", model.star_age)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(
        out,
        "          initial_z      {:26.16E}",
        model.controls.initial_z
    )
    .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "           n_shells      {:26}", model.nz)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "           net_name   '{}'", model.job.net_name)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "            species      {:26}", model.species)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "               Teff      {:26.16E}", model.teff)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "   year_for_time_step      {dt_yrs:26.16E}")
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out).map_err(|e| StarError::IoError(e.to_string()))?;

    write!(
        out,
        "        lnd                       lnT                       lnR                       L                         dq"
    )
    .map_err(|e| StarError::IoError(e.to_string()))?;
    for name in &model.iso_names {
        write!(out, "  {name:>24}")
            .map_err(|e| StarError::IoError(e.to_string()))?;
    }
    writeln!(out).map_err(|e| StarError::IoError(e.to_string()))?;

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
        .map_err(|e| StarError::IoError(e.to_string()))?;
        let xa_k = model.cell_xa(k);
        for &x_j in xa_k {
            write!(out, "  {x_j:24.16E}")
                .map_err(|e| StarError::IoError(e.to_string()))?;
        }
        writeln!(out).map_err(|e| StarError::IoError(e.to_string()))?;
    }

    writeln!(out).map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "        previous model")
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "   previous n_shells      {:26}", model.nz)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "  previous mass (grams)   {:26.16E}", model.mstar_old)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "   timestep (seconds)     {:26.16E}", model.dt_old)
        .map_err(|e| StarError::IoError(e.to_string()))?;
    writeln!(out, "    dt_next (seconds)     {:26.16E}", model.dt_next)
        .map_err(|e| StarError::IoError(e.to_string()))?;

    Ok(out)
}

/// Parses an upstream MESA ASCII `.mod` string into a [`StarModel`]
/// (`read_model.f90`).
///
/// # Errors
/// Returns [`StarError::IoError`] if the `.mod` text is malformed.
pub fn read_mod_str(
    text: &str,
    mut controls: StarControls,
    mut job: StarJobControls,
) -> Result<StarModel, StarError> {
    let mut lines = text.lines().peekable();
    let mut m_msun = controls.initial_mass;
    let mut model_number = 1_usize;
    let mut star_age = 0.0_f64;
    let mut nz = 0_usize;
    let mut teff = 5777.0_f64;
    let mut dt_yrs = 1000.0_f64;
    let mut header_cols: Vec<String> = Vec::new();

    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('!') {
            continue;
        }
        if trimmed.starts_with("M/Msun") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                m_msun = parse_fortran_f64(v)?;
                controls.initial_mass = m_msun;
            }
        } else if trimmed.starts_with("model_number") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                model_number = v.parse::<usize>().unwrap_or(1);
            }
        } else if trimmed.starts_with("star_age") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                star_age = parse_fortran_f64(v)?;
            }
        } else if trimmed.starts_with("initial_z")
            || trimmed.starts_with("initial_Z")
        {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                controls.initial_z = parse_fortran_f64(v)?;
            }
        } else if trimmed.starts_with("n_shells") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                nz = v.parse::<usize>().map_err(|e| {
                    StarError::IoError(format!("invalid n_shells: {e}"))
                })?;
            }
        } else if trimmed.starts_with("net_name") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                job.net_name =
                    v.trim_matches('\'').trim_matches('"').to_string();
            }
        } else if trimmed.starts_with("Teff") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                teff = parse_fortran_f64(v)?;
            }
        } else if trimmed.starts_with("year_for_time_step") {
            if let Some(v) = trimmed.split_whitespace().nth(1) {
                dt_yrs = parse_fortran_f64(v)?;
            }
        } else if trimmed.starts_with("lnd ")
            || (trimmed.contains("lnd") && trimmed.contains("lnT"))
        {
            header_cols = trimmed
                .split_whitespace()
                .map(ToString::to_string)
                .collect();
            break;
        }
    }

    if nz == 0 || header_cols.len() < 5 {
        return Err(StarError::IoError(String::from(
            "missing n_shells or column header in .mod file",
        )));
    }
    let lnd_idx =
        header_cols.iter().position(|c| c == "lnd").ok_or_else(|| {
            StarError::IoError(String::from("missing lnd column"))
        })?;
    // Determine whether v or omega columns precede dq
    let dq_idx = header_cols
        .iter()
        .position(|c| c == "dq")
        .ok_or_else(|| StarError::IoError(String::from("missing dq column")))?;
    let iso_names = &header_cols[dq_idx + 1..];
    let species = iso_names.len();
    let catalog = NuclideCatalog::builtin_ref();
    let chem_id: Vec<usize> = iso_names
        .iter()
        .map(|n| catalog.get_nuclide_index(n).unwrap_or(1))
        .collect();

    let mut model = StarModel::new(nz, chem_id, job, controls);
    model.mstar = m_msun * M_SUN;
    model.mstar_start = model.mstar;
    model.mstar_old = model.mstar;
    model.model_number = model_number;
    model.star_age = star_age;
    model.teff = teff;
    model.dt = dt_yrs * SECYER;
    model.dt_next = model.dt;
    model.dt_old = model.dt;

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
        if toks.len() < 1 + (dq_idx - lnd_idx + 1) + species {
            return Err(StarError::IoError(format!(
                "short zone row in .mod at zone {}: '{}'",
                k + 1,
                line
            )));
        }
        // First token is 1-based zone index k+1
        let offset = 1;
        model.lnd[k] = parse_fortran_f64(toks[offset])?;
        model.ln_t[k] = parse_fortran_f64(toks[offset + 1])?;
        model.ln_r[k] = parse_fortran_f64(toks[offset + 2])?;
        model.lum[k] = parse_fortran_f64(toks[offset + 3])?;
        let dq_col = offset + (dq_idx - lnd_idx);
        model.dq[k] = parse_fortran_f64(toks[dq_col])?;
        for j in 0..species {
            model.xa[k * species + j] =
                parse_fortran_f64(toks[dq_col + 1 + j])?;
        }
        k += 1;
    }

    // Parse optional footer scalars
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
                parsed_dt_old = parse_fortran_f64(v)?;
            }
        } else if t.starts_with("dt_next (seconds)") {
            if let Some(v) = t.split_whitespace().last() {
                model.dt_next = parse_fortran_f64(v)?;
            }
        }
    }

    model.finish_init_model()?;
    model.mstar_old = parsed_mstar_old;
    model.dt_old = parsed_dt_old;

    Ok(model)
}

/// Serializes a [`StarModel`] into binary `.photo` restart bytes with
/// `STAR_DEF_VERSION = 19` and `part_number = 1..7` sentinels (`photo_out.f90`).
#[must_use]
pub fn write_photo_bytes(model: &StarModel) -> Vec<u8> {
    let mut buf =
        Vec::with_capacity(model.nz * (8 * (10 + model.species)) + 256);
    let write_i32 = |buf: &mut Vec<u8>, v: i32| {
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let write_u64 = |buf: &mut Vec<u8>, v: u64| {
        buf.extend_from_slice(&v.to_le_bytes());
    };
    let write_f64 = |buf: &mut Vec<u8>, v: f64| {
        buf.extend_from_slice(&v.to_le_bytes());
    };
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

    // Part 2: primary hydro arrays (`lnd`, `ln_t`, `ln_r`, `lum`, `v`, `u`, `dq`, `xa`)
    write_i32(&mut buf, 2);
    write_f64_slice(&mut buf, &model.lnd);
    write_f64_slice(&mut buf, &model.ln_t);
    write_f64_slice(&mut buf, &model.ln_r);
    write_f64_slice(&mut buf, &model.lum);
    write_f64_slice(&mut buf, &model.v);
    write_f64_slice(&mut buf, &model.u);
    write_f64_slice(&mut buf, &model.dq);
    write_f64_slice(&mut buf, &model.xa);

    // Part 3: old-step arrays
    write_i32(&mut buf, 3);
    write_f64_slice(&mut buf, &model.lnd_old);
    write_f64_slice(&mut buf, &model.ln_t_old);
    write_f64_slice(&mut buf, &model.ln_r_old);
    write_f64_slice(&mut buf, &model.lum_old);
    write_f64_slice(&mut buf, &model.dq_old);
    write_f64_slice(&mut buf, &model.xa_old);

    // Part 4: evolutionary scalar state
    write_i32(&mut buf, 4);
    write_u64(&mut buf, model.model_number as u64);
    write_f64(&mut buf, model.star_age);
    write_f64(&mut buf, model.star_age_old);
    write_f64(&mut buf, model.dt);
    write_f64(&mut buf, model.dt_next);
    write_f64(&mut buf, model.dt_old);
    write_f64(&mut buf, model.mstar);
    write_f64(&mut buf, model.mstar_old);
    write_f64(&mut buf, model.mstar_dot);
    write_f64(&mut buf, model.m_center);
    write_f64(&mut buf, model.r_center);
    write_f64(&mut buf, model.l_center);
    write_f64(&mut buf, model.v_center);
    write_f64(&mut buf, model.teff);
    write_f64(&mut buf, model.tau_base);
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
    let read_i32 = |pos: &mut usize| -> Result<i32, StarError> {
        if *pos + 4 > bytes.len() {
            return Err(StarError::IoError(String::from(
                "unexpected EOF reading i32 in .photo",
            )));
        }
        let v = i32::from_le_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let read_u64 = |pos: &mut usize| -> Result<u64, StarError> {
        if *pos + 8 > bytes.len() {
            return Err(StarError::IoError(String::from(
                "unexpected EOF reading u64 in .photo",
            )));
        }
        let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(v)
    };
    let read_f64 = |pos: &mut usize| -> Result<f64, StarError> {
        if *pos + 8 > bytes.len() {
            return Err(StarError::IoError(String::from(
                "unexpected EOF reading f64 in .photo",
            )));
        }
        let v = f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(v)
    };
    let read_f64_vec = |pos: &mut usize,
                        n: usize|
     -> Result<Vec<f64>, StarError> {
        if *pos + 8 * n > bytes.len() {
            return Err(StarError::IoError(String::from(
                "unexpected EOF reading f64 slice in .photo",
            )));
        }
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let x =
                f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            v.push(x);
        }
        Ok(v)
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

    expect_part(&mut pos, 2)?;
    model.lnd = read_f64_vec(&mut pos, nz)?;
    model.ln_t = read_f64_vec(&mut pos, nz)?;
    model.ln_r = read_f64_vec(&mut pos, nz)?;
    model.lum = read_f64_vec(&mut pos, nz)?;
    model.v = read_f64_vec(&mut pos, nz)?;
    model.u = read_f64_vec(&mut pos, nz)?;
    model.dq = read_f64_vec(&mut pos, nz)?;
    model.xa = read_f64_vec(&mut pos, nz * species)?;

    expect_part(&mut pos, 3)?;
    model.lnd_old = read_f64_vec(&mut pos, nz)?;
    model.ln_t_old = read_f64_vec(&mut pos, nz)?;
    model.ln_r_old = read_f64_vec(&mut pos, nz)?;
    model.lum_old = read_f64_vec(&mut pos, nz)?;
    model.dq_old = read_f64_vec(&mut pos, nz)?;
    model.xa_old = read_f64_vec(&mut pos, nz * species)?;

    expect_part(&mut pos, 4)?;
    model.model_number = read_u64(&mut pos)? as usize;
    model.star_age = read_f64(&mut pos)?;
    model.star_age_old = read_f64(&mut pos)?;
    model.dt = read_f64(&mut pos)?;
    model.dt_next = read_f64(&mut pos)?;
    model.dt_old = read_f64(&mut pos)?;
    model.mstar = read_f64(&mut pos)?;
    model.mstar_old = read_f64(&mut pos)?;
    model.mstar_dot = read_f64(&mut pos)?;
    model.m_center = read_f64(&mut pos)?;
    model.r_center = read_f64(&mut pos)?;
    model.l_center = read_f64(&mut pos)?;
    model.v_center = read_f64(&mut pos)?;
    model.teff = read_f64(&mut pos)?;
    model.tau_base = read_f64(&mut pos)?;
    model.total_num_solver_iterations = read_u64(&mut pos)? as usize;
    model.num_retries = read_u64(&mut pos)? as usize;

    expect_part(&mut pos, 5)?;
    expect_part(&mut pos, 6)?;
    expect_part(&mut pos, 7)?;

    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.set_m_grav_and_grav();
    model.pack_xh();
    model.save_start_values();

    Ok(model)
}
