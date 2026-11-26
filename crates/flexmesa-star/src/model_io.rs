// SPDX-License-Identifier: GPL-3.0-or-later
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
/// `RSP` with a stored previous model), core boundary headers, and `RSP` /
/// `RSP2` zone columns.
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
        && (write_rsp
            || model.aux.have_mlt_vc
            || model.mlt_vc.iter().any(|&x| x != 0.0));

    // `write_model.f90:74-77`: `prev_flag = (s% nz_old == s% nz .and.
    // s% generations > 1)` sets `bit_for_2models` (2) and gates the
    // "previous model" block (`:218-231`). `dq_old` is copied with the other
    // old-model vectors when upstream sets `s% nz_old = s% nz`
    // (`evolve_support.f90:81`), so its length stands for `nz_old`.
    let prev_flag = model.generations > 1 && model.dq_old.len() == nz;
    let mut file_type = 0_u32;
    if prev_flag {
        file_type |= 4;
    }
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
    writeln!(
        out,
        "! note: initial lines of file can contain comments\n! flexmesa math_backend: {}\n!",
        flexmesa_math::numerics_mode().as_str()
    )
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
        (
            "crystal_core_boundary_mass",
            model.aux.crystal_core_boundary_mass,
            model.aux.crystal_core_boundary_mass > 0.0,
        ),
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

    if !prev_flag {
        return Ok(out);
    }
    let dt_next_to_write = if model.aux.step_hist.dt_next_unclipped > 0.0 {
        model.aux.step_hist.dt_next_unclipped
    } else {
        model.dt_next
    };
    // `write_model.f90:223-229`: `timestep (seconds)` records `s% dt`, not `dt_old`.
    let (m_old, dt_cur, dt_next) =
        (model.mstar_old, model.dt, dt_next_to_write);
    writeln!(out, "\n        previous model").unwrap();
    writeln!(out, "   previous n_shells      {nz:26}").unwrap();
    writeln!(out, "  previous mass (grams)   {m_old:26.16E}").unwrap();
    writeln!(out, "   timestep (seconds)     {dt_cur:26.16E}").unwrap();
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
    controls: StarControls,
    job: StarJobControls,
) -> Result<StarModel, StarError> {
    read_mod_str_with_dir(text, controls, job, None)
}

/// Parses a MESA `.mod` text file (`star/private/read_model.f90`), resolving
/// any non-empty network name against `search_dir` before falling back to the
/// built-in network bundle (`net/private/net_initialize.f90:286-302`).
///
/// # Errors
/// Returns [`StarError`] if headers, zone columns, or the requested network
/// fail validation.
pub fn read_mod_str_with_dir(
    text: &str,
    mut controls: StarControls,
    mut job: StarJobControls,
    search_dir: Option<&std::path::Path>,
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
    let mut crystal_core_boundary_mass = -1.0_f64;
    let mut header_cols: Vec<String> = Vec::new();
    let mut file_type_opt: Option<u32> = None;

    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('!') {
            continue;
        }
        if file_type_opt.is_none() {
            let first_tok = trimmed.split_whitespace().next().unwrap_or("");
            if let Ok(ft) = first_tok.parse::<u32>() {
                file_type_opt = Some(ft);
                // Reject obsolete lnPgas-variable models (`read_model.f90:264-270`).
                if (ft & (1 << 1)) != 0 {
                    return Err(StarError::IoError(String::from(
                        "MESA no longer supports models using lnPgas as a \
                         structure variable (`star/private/read_model.f90:264-270`)",
                    )));
                }
                continue;
            }
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
            "model_number" => {
                model_number = v.parse::<usize>().map_err(|e| {
                    StarError::IoError(format!(
                        "invalid .mod model_number '{v}': {e}"
                    ))
                })?;
            }
            "star_age" => star_age = parse_fortran_f64(v)?,
            "initial_z" | "initial_Z" => {
                controls.initial_z = parse_fortran_f64(v)?;
            }
            "initial_y" | "initial_Y" => {
                controls.initial_y = parse_fortran_f64(v)?;
            }
            "n_shells" => {
                nz = v.parse::<usize>().map_err(|e| {
                    StarError::IoError(format!(
                        "invalid .mod n_shells '{v}': {e}"
                    ))
                })?;
            }
            "net_name" => {
                job.net_name = v.trim_matches(['\'', '"']).to_string();
            }
            "species" => {
                let sp = v.parse::<usize>().map_err(|e| {
                    StarError::IoError(format!(
                        "invalid .mod species '{v}': {e}"
                    ))
                })?;
                if sp == 0 {
                    return Err(StarError::IoError(String::from(
                        "invalid .mod species 0",
                    )));
                }
                header_species = Some(sp);
            }
            "Teff" => teff = parse_fortran_f64(v)?,
            "year_for_time_step" => dt_yrs = Some(parse_fortran_f64(v)?),
            "xmstar" => xmstar_opt = Some(parse_fortran_f64(v)?),
            "M_center" => m_center_opt = Some(parse_fortran_f64(v)?),
            "R_center" => r_center = parse_fortran_f64(v)?,
            "L_center" => l_center = parse_fortran_f64(v)?,
            "v_center" => v_center = parse_fortran_f64(v)?,
            "total_energy" => total_energy = parse_fortran_f64(v)?,
            "cumulative_energy_error" => cum_energy_err = parse_fortran_f64(v)?,
            "tau_factor" => {
                // A saved `tau_factor` replaces the inlist value unless the
                // inlist just set it (`star/private/read_model.f90:307-315`).
                let saved = parse_fortran_f64(v)?;
                if (saved - controls.tau_factor).abs() > saved * 1.0e-9
                    && controls.tau_factor != job.set_to_this_tau_factor
                {
                    controls.tau_factor = saved;
                }
            }
            "opacity_factor" => controls.opacity_factor = parse_fortran_f64(v)?,
            "crystal_core_boundary_mass" => {
                crystal_core_boundary_mass = parse_fortran_f64(v)?;
            }
            "num_retries" => {
                num_retries = v.parse::<usize>().map_err(|e| {
                    StarError::IoError(format!(
                        "invalid .mod num_retries '{v}': {e}"
                    ))
                })?;
            }
            _ => {}
        }
    }

    if nz == 0
        || header_cols.len() < 5
        || !m_msun.is_finite()
        || m_msun < 0.0
        || !controls.initial_z.is_finite()
        || controls.initial_z < 0.0
    {
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
    let col_chem_id: Vec<usize> = iso_names
        .iter()
        .map(|n| {
            catalog.get_nuclide_index(n).ok_or_else(|| {
                StarError::IoError(format!(
                    "unknown isotope '{n}' in .mod header (`star/private/read_model.f90:414-420`)"
                ))
            })
        })
        .collect::<Result<Vec<usize>, StarError>>()?;

    // `do_read_saved_model` loads the model's own net (`set_net`,
    // `star/private/read_model.f90:364,381`), so species follow the net's
    // order and column `i` is stored at `perm(i) = net_iso(chem_id)`
    // (`get_chem_col_names`, `star/private/read_model.f90:918-984`;
    // `xa(perm(ii),k) = vec(j+ii)`, `star/private/read_model.f90:648-650`).
    // A column the net lacks stops the run (`read_model.f90:412-420`). A
    // change_net / change_initial_net is applied later, after
    // `finish_load_model`, by `do_star_job_controls_after`
    // (`star/job/run_star_support.f90:1912-1916`).
    let (chem_id, perm): (Vec<usize>, Vec<usize>) = if job
        .net_name
        .trim()
        .is_empty()
    {
        (col_chem_id.clone(), (0..species).collect())
    } else {
        let net = crate::micro::get_or_load_net(&job.net_name, search_dir)?;
        let net_chem_id = net.chem_id.clone();
        let mut perm = Vec::with_capacity(species);
        for (n, cid) in iso_names.iter().zip(&col_chem_id) {
            let j = net_chem_id.iter().position(|c| c == cid).ok_or_else(
                    || {
                        StarError::IoError(format!(
                            "Mod file has isotope {n} but that is not in the net {} (`star/private/read_model.f90:412-420`)",
                            job.net_name
                        ))
                    },
                )?;
            perm.push(j);
        }
        // DECISIONS #43, deferred #472: with fewer columns than net species upstream
        // leaves the remaining `xa` rows unset (`set_net` sets
        // `s% species` from the net, `star/private/net.f90:626`, while
        // only `species` columns are read).
        if net_chem_id.len() != species {
            return Err(StarError::IoError(format!(
                    ".mod has {species} isotope columns but net {} has {} species; upstream reads unset abundances here (`star/private/read_model.f90:405-420`, `star/private/net.f90:626`; DECISIONS #43, deferred #472)",
                    job.net_name,
                    net_chem_id.len()
                )));
        }
        (net_chem_id, perm)
    };

    let mut model = StarModel::new(nz, chem_id, job.clone(), controls);
    model.aux.have_j_rot = j_rot_idx.is_some();
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
    // `star/defaults/star_job.defaults:633-634` and
    // `star/job/run_star_support.f90:2009-2012`: `set_initial_number_retries`
    // defaults to `true`, resetting `num_retries` to `initial_number_retries`
    // (`0`) on `.mod` load unless explicitly disabled.
    let eff_num_retries = if model.job.set_initial_number_retries {
        model.job.initial_number_retries
    } else {
        num_retries
    };
    (model.model_number, model.num_retries) = (model_number, eff_num_retries);
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
        for (j, &p) in perm.iter().enumerate() {
            model.xa[k * species + p] =
                parse_fortran_f64(toks[offset + iso_start_idx + j])?;
        }
        k += 1;
    }

    let (mut parsed_mstar_old, mut parsed_dt_old) =
        (model.mstar_old, model.dt_old);
    // `read_model.f90:433-436`: the previous-model properties are read only
    // when the header sets `bit_for_2models` (2); otherwise any trailing block
    // is ignored.
    let do_read_prev = file_type_opt.is_some_and(|ft| ft & (1 << 2) != 0);
    let (mut prev_mass_opt, mut prev_dt_opt, mut prev_dt_next_opt) =
        (None, None, None);
    for line in lines.filter(|_| do_read_prev) {
        let t = line.trim();
        if t.starts_with("previous model") {
            continue;
        }
        let Some(v) = t.split_whitespace().last() else {
            continue;
        };
        if t.starts_with("previous mass (grams)") {
            let m_old = parse_fortran_f64(v)?;
            parsed_mstar_old = m_old;
            prev_mass_opt = Some(m_old);
        } else if t.starts_with("timestep (seconds)") {
            // `read_model.f90:490`: `timestep (seconds)` unconditionally sets `s% dt`.
            let dt_sec = parse_fortran_f64(v)?;
            parsed_dt_old = dt_sec;
            model.dt = dt_sec;
            prev_dt_opt = Some(dt_sec);
        } else if t.starts_with("dt_next (seconds)") {
            let dt_next_sec = parse_fortran_f64(v)?;
            model.dt_next = dt_next_sec;
            prev_dt_next_opt = Some(dt_next_sec);
        }
    }
    if do_read_prev {
        let missing_or_neg = |opt: Option<f64>| !matches!(opt, Some(v) if v.is_finite() && v >= 0.0);
        if missing_or_neg(prev_dt_opt) {
            return Err(StarError::IoError(String::from(
                "missing or negative dt for previous model \
                 (`star/private/read_model.f90:505-508`)",
            )));
        }
        if missing_or_neg(prev_mass_opt) {
            return Err(StarError::IoError(String::from(
                "missing or negative mstar_old for previous model \
                 (`star/private/read_model.f90:509-512`)",
            )));
        }
        if missing_or_neg(prev_dt_next_opt) {
            return Err(StarError::IoError(String::from(
                "missing or negative dt_next for previous model \
                 (`star/private/read_model.f90:513-516`)",
            )));
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
        model.save_start_values()?;
        model.save_old_values();
        // `star/private/read_model.f90:438, 463`.
        model.generations = 1;
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
        model.aux.have_mlt_vc = true;
        // Upstream `init.f90:541` + `read_model.f90:157-162`: keep `.mod`
        // `mlt_vc` intact across startup `set_vars` until `set_current_to_old`.
        model.aux.okay_to_set_mlt_vc = false;
    } else {
        model.aux.okay_to_set_mlt_vc = true;
    }
    if let Some(hp) = saved_hp {
        model.scale_height.clone_from(&hp);
        model.aux.rsp.hp_face = hp;
    }
    (model.mstar_old, model.dt_old) = (parsed_mstar_old, parsed_dt_old);
    model.init_model_number = model.model_number as isize;
    model.time_sec = model.star_age * flexmesa_const::SECYER;
    model.time_old_sec = model.star_age_old * flexmesa_const::SECYER;
    model.aux.step_hist.time_sec = model.time_sec;
    model.aux.step_hist.time_old_sec = model.time_old_sec;
    model.m_center_old = model.m_center;
    model.xmstar_old = model.xmstar;
    model.r_center_old = model.r_center;
    model.l_center_old = model.l_center;
    model.v_center_old = model.v_center;
    model.aux.crystal_core_boundary_mass = crystal_core_boundary_mass;

    Ok(model)
}

pub use crate::photo_io::{
    compute_photo_checksum, load_photo_bytes, photo_numerics_mode,
    read_photo_bytes, try_write_photo_bytes, write_photo_bytes,
    PHOTO_FORMAT_VERSION,
};
