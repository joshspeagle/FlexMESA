// SPDX-License-Identifier: LGPL-3.0-or-later
//! Binary `.photo` restart checkpoint serializer, deserializer, and 64-bit
//! FNV-1a checksum (`photo_in.f90`, `photo_out.f90`, and `./ck`).

use flexmesa_kap::{KapCoOption, KapLowTOption, KapOption};

use crate::controls::{StarControls, StarError, StarJobControls};
use crate::model::StarModel;
use crate::model_io::STAR_DEF_VERSION;

fn kap_option_to_u8(opt: KapOption) -> u8 {
    match opt {
        KapOption::Gn93 => 0,
        KapOption::Gs98 => 1,
        KapOption::A09 => 2,
        KapOption::OpGs98 => 3,
        KapOption::OpA09 => 4,
    }
}

fn kap_option_from_u8(v: u8) -> Result<KapOption, StarError> {
    match v {
        0 => Ok(KapOption::Gn93),
        1 => Ok(KapOption::Gs98),
        2 => Ok(KapOption::A09),
        3 => Ok(KapOption::OpGs98),
        4 => Ok(KapOption::OpA09),
        _ => Err(StarError::IoError(format!("invalid KapOption tag {v}"))),
    }
}

fn kap_co_option_to_u8(opt: KapCoOption) -> u8 {
    match opt {
        KapCoOption::Gn93Co => 0,
        KapCoOption::Gs98Co => 1,
        KapCoOption::A09Co => 2,
    }
}

fn kap_co_option_from_u8(v: u8) -> Result<KapCoOption, StarError> {
    match v {
        0 => Ok(KapCoOption::Gn93Co),
        1 => Ok(KapCoOption::Gs98Co),
        2 => Ok(KapCoOption::A09Co),
        _ => Err(StarError::IoError(format!("invalid KapCoOption tag {v}"))),
    }
}

fn kap_low_t_option_to_u8(opt: KapLowTOption) -> u8 {
    match opt {
        KapLowTOption::LowTFa05Gs98 => 0,
        KapLowTOption::LowTFa05Gn93 => 1,
        KapLowTOption::LowTFa05A09p => 2,
        KapLowTOption::LowTFreedman11 => 3,
        KapLowTOption::KapCn => 4,
        KapLowTOption::Aesopus => 5,
    }
}

fn kap_low_t_option_from_u8(v: u8) -> Result<KapLowTOption, StarError> {
    match v {
        0 => Ok(KapLowTOption::LowTFa05Gs98),
        1 => Ok(KapLowTOption::LowTFa05Gn93),
        2 => Ok(KapLowTOption::LowTFa05A09p),
        3 => Ok(KapLowTOption::LowTFreedman11),
        4 => Ok(KapLowTOption::KapCn),
        5 => Ok(KapLowTOption::Aesopus),
        _ => Err(StarError::IoError(format!("invalid KapLowTOption tag {v}"))),
    }
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
        Vec::with_capacity(model.nz * (8 * (12 + model.species)) + 384);
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

    write_i32(&mut buf, 1);
    write_u64(&mut buf, model.nz as u64);
    write_u64(&mut buf, model.species as u64);
    write_u64(&mut buf, model.layout.nvar_hydro as u64);
    for &cid in &model.chem_id {
        write_u64(&mut buf, cid as u64);
    }
    let net_bytes = model.job.net_name.as_bytes();
    write_u64(&mut buf, net_bytes.len() as u64);
    buf.extend_from_slice(net_bytes);
    buf.extend_from_slice(&[
        u8::from(model.controls.v_flag),
        u8::from(model.controls.u_flag),
        u8::from(model.controls.rotation_flag),
        u8::from(model.controls.op_split_burn),
        u8::from(model.controls.rsp_flag),
        u8::from(model.controls.rsp2_flag),
        u8::from(model.controls.rti_flag),
    ]);

    write_i32(&mut buf, 2);
    for s in [&model.lnd, &model.ln_t, &model.ln_r, &model.lum] {
        write_f64_slice(&mut buf, s);
    }
    for s in [&model.v, &model.u, &model.dq, &model.xa] {
        write_f64_slice(&mut buf, s);
    }
    for s in [&model.omega, &model.j_rot] {
        write_u64(&mut buf, s.len() as u64);
        write_f64_slice(&mut buf, s);
    }

    write_i32(&mut buf, 3);
    write_u64(&mut buf, model.lnd_old.len() as u64);
    for s in [
        &model.lnd_old,
        &model.ln_t_old,
        &model.ln_r_old,
        &model.lum_old,
        &model.v_old,
        &model.u_old,
        &model.dq_old,
        &model.aux.alpha_rti_old,
        &model.xa_old,
    ] {
        write_f64_slice(&mut buf, s);
    }
    write_u64(&mut buf, model.lnd_start.len() as u64);
    for s in [
        &model.lnd_start,
        &model.ln_t_start,
        &model.ln_r_start,
        &model.lum_start,
        &model.v_start,
        &model.u_start,
    ] {
        write_f64_slice(&mut buf, s);
    }

    write_i32(&mut buf, 4);
    write_u64(&mut buf, model.model_number as u64);
    write_f64_slice(
        &mut buf,
        &[
            model.star_age,
            model.star_age_old,
            model.dt,
            model.dt_next,
            model.dt_old,
            model.dt_limit_ratio_old,
            model.mstar,
            model.mstar_old,
            model.mstar_dot,
            model.m_center,
            model.r_center,
            model.l_center,
            model.v_center,
            model.teff,
            model.tau_base,
            model.mstar_start,
            model.aux.total_energy,
            model.aux.cumulative_energy_error,
        ],
    );
    write_u64(&mut buf, model.total_num_solver_iterations as u64);
    write_u64(&mut buf, model.num_retries as u64);

    write_i32(&mut buf, 5);
    for s in [
        &model.eps_mdot,
        &model.cgrav,
        &model.aux.irradiation_heat,
        &model.aux.etrb,
        &model.aux.alpha_rti,
        &model.aux.w_div_wc,
        &model.mlt_vc,
        &model.aux.mlt_vc_old,
        &model.aux.burn_avg_epsnuc,
        &model.sig,
    ] {
        write_f64_slice(&mut buf, s);
    }

    let ec = &model.eos_config;
    let kc = &model.kap_config;
    buf.extend_from_slice(&[
        u8::from(ec.use_cms),
        u8::from(ec.use_skye),
        u8::from(ec.use_pc),
        u8::from(ec.use_free_eos),
        u8::from(ec.use_opal_scvh),
        u8::from(ec.okay_to_convert_ierr_to_skip),
        u8::from(ec.linear_blend),
        kap_option_to_u8(kc.kap_option),
        kap_co_option_to_u8(kc.kap_co_option),
        kap_low_t_option_to_u8(kc.kap_low_t_option),
        u8::from(kc.cubic_interpolation_in_x),
        u8::from(kc.cubic_interpolation_in_z),
        u8::from(kc.include_electron_conduction),
        u8::from(kc.use_blouin_conductive_opacities),
        u8::from(kc.use_zbase_for_type1),
        u8::from(kc.use_type2_opacities),
    ]);
    write_f64_slice(
        &mut buf,
        &[
            ec.helm_blend_width,
            kc.zbase,
            kc.kap_blend_log_t_lower_bdy,
            kc.kap_blend_log_t_upper_bdy,
            kc.kap_type2_full_on_x,
            kc.kap_type2_full_off_x,
            kc.kap_type2_full_off_dz,
            kc.kap_type2_full_on_dz,
            kc.log_t_compton_blend_hi,
            kc.log_r_compton_blend_lo,
        ],
    );

    write_i32(&mut buf, 6);
    let r = &model.aux.rsp;
    let has_rsp = model.controls.rsp_flag
        || model.controls.rsp2_flag
        || r.num_periods > 0
        || r.have_set_velocities
        || r.et_rsp.iter().any(|&x| x != 0.0)
        || r.erad_rsp.iter().any(|&x| x != 0.0)
        || r.hp_face.iter().any(|&x| x != 0.0);
    buf.push(u8::from(has_rsp));
    if has_rsp {
        for s in r.zone_vecs() {
            write_f64_slice(&mut buf, s);
        }
        write_f64_slice(&mut buf, &r.pack_scalars());
        if model.controls.rsp_flag {
            for s in [
                &model.t,
                &model.pgas,
                &model.prad,
                &model.opacity,
                &model.csound,
                &model.cp,
                &model.r,
                &model.m,
                &model.dm,
                &model.dm_bar,
            ] {
                write_f64_slice(&mut buf, s);
            }
        }
    }
    write_i32(&mut buf, 7);
    buf
}

/// Restores a [`StarModel`] from binary `.photo` restart bytes (`photo_in.f90`).
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
            (p == expected).then_some(()).ok_or_else(|| {
                StarError::IoError(format!(
                    "photo sentinel mismatch: expected part {expected}, got {p}"
                ))
            })
        };

    let version = read_i32(&mut pos)?;
    if version != STAR_DEF_VERSION {
        return Err(StarError::IoError(format!(
            "unsupported photo version {version} (expected {STAR_DEF_VERSION})"
        )));
    }

    expect_part(&mut pos, 1)?;
    let (nz, species) =
        (read_u64(&mut pos)? as usize, read_u64(&mut pos)? as usize);
    let _nvar_hydro = read_u64(&mut pos)? as usize;
    let mut chem_id = Vec::with_capacity(species);
    for _ in 0..species {
        chem_id.push(read_u64(&mut pos)? as usize);
    }
    let net_len = read_u64(&mut pos)? as usize;
    let net_slice = bytes
        .get(pos..pos + net_len)
        .ok_or_else(|| eof("EOF net_name"))?;
    pos += net_len;
    let saved_net_name = std::str::from_utf8(net_slice)
        .map_err(|e| {
            StarError::IoError(format!("invalid UTF-8 net_name: {e}"))
        })?
        .to_owned();
    let job_tags = bytes
        .get(pos..pos + 7)
        .ok_or_else(|| eof("EOF job/controls flags"))?;
    pos += 7;
    let mut job = job;
    let mut controls = controls;
    job.net_name = saved_net_name;
    controls.v_flag = job_tags[0] != 0;
    controls.u_flag = job_tags[1] != 0;
    controls.rotation_flag = job_tags[2] != 0;
    controls.op_split_burn = job_tags[3] != 0;
    controls.rsp_flag = job_tags[4] != 0;
    controls.rsp2_flag = job_tags[5] != 0;
    controls.rti_flag = job_tags[6] != 0;

    let mut model = StarModel::new(nz, chem_id, job, controls);

    expect_part(&mut pos, 2)?;
    for dst in [
        &mut model.lnd,
        &mut model.ln_t,
        &mut model.ln_r,
        &mut model.lum,
        &mut model.v,
        &mut model.u,
        &mut model.dq,
    ] {
        *dst = read_f64_vec(&mut pos, nz)?;
    }
    model.xa = read_f64_vec(&mut pos, nz * species)?;
    let n_omega = read_u64(&mut pos)? as usize;
    model.omega = read_f64_vec(&mut pos, n_omega)?;
    let n_jrot = read_u64(&mut pos)? as usize;
    model.j_rot = read_f64_vec(&mut pos, n_jrot)?;

    expect_part(&mut pos, 3)?;
    let nz_old = read_u64(&mut pos)? as usize;
    for dst in [
        &mut model.lnd_old,
        &mut model.ln_t_old,
        &mut model.ln_r_old,
        &mut model.lum_old,
        &mut model.v_old,
        &mut model.u_old,
        &mut model.dq_old,
        &mut model.aux.alpha_rti_old,
    ] {
        *dst = read_f64_vec(&mut pos, nz_old)?;
    }
    model.xa_old = read_f64_vec(&mut pos, nz_old * species)?;
    let nz_start = read_u64(&mut pos)? as usize;
    let saved_lnd_start = read_f64_vec(&mut pos, nz_start)?;
    let saved_ln_t_start = read_f64_vec(&mut pos, nz_start)?;
    let saved_ln_r_start = read_f64_vec(&mut pos, nz_start)?;
    let saved_lum_start = read_f64_vec(&mut pos, nz_start)?;
    let saved_v_start = read_f64_vec(&mut pos, nz_start)?;
    let saved_u_start = read_f64_vec(&mut pos, nz_start)?;

    expect_part(&mut pos, 4)?;
    model.model_number = read_u64(&mut pos)? as usize;
    let s = read_f64_vec(&mut pos, 18)?;
    (model.star_age, model.star_age_old, model.dt) = (s[0], s[1], s[2]);
    (model.dt_next, model.dt_old, model.dt_limit_ratio_old) =
        (s[3], s[4], s[5]);
    (model.mstar, model.mstar_old, model.mstar_dot) = (s[6], s[7], s[8]);
    (model.m_center, model.r_center, model.l_center) = (s[9], s[10], s[11]);
    (model.v_center, model.teff, model.tau_base) = (s[12], s[13], s[14]);
    let saved_mstar_start = s[15];
    model.aux.total_energy = s[16];
    model.aux.cumulative_energy_error = s[17];
    model.total_num_solver_iterations = read_u64(&mut pos)? as usize;
    model.num_retries = read_u64(&mut pos)? as usize;

    expect_part(&mut pos, 5)?;
    for dst in [
        &mut model.eps_mdot,
        &mut model.cgrav,
        &mut model.aux.irradiation_heat,
        &mut model.aux.etrb,
        &mut model.aux.alpha_rti,
        &mut model.aux.w_div_wc,
    ] {
        *dst = read_f64_vec(&mut pos, nz)?;
    }
    let saved_mlt_vc = read_f64_vec(&mut pos, nz)?;
    let saved_mlt_vc_old = read_f64_vec(&mut pos, nz)?;
    model.aux.burn_avg_epsnuc = read_f64_vec(&mut pos, nz)?;
    let saved_sig = read_f64_vec(&mut pos, nz)?;

    let cfg_tags = bytes
        .get(pos..pos + 16)
        .ok_or_else(|| eof("EOF eos/kap tags"))?;
    pos += 16;
    model.eos_config.use_cms = cfg_tags[0] != 0;
    model.eos_config.use_skye = cfg_tags[1] != 0;
    model.eos_config.use_pc = cfg_tags[2] != 0;
    model.eos_config.use_free_eos = cfg_tags[3] != 0;
    model.eos_config.use_opal_scvh = cfg_tags[4] != 0;
    model.eos_config.okay_to_convert_ierr_to_skip = cfg_tags[5] != 0;
    model.eos_config.linear_blend = cfg_tags[6] != 0;
    model.kap_config.kap_option = kap_option_from_u8(cfg_tags[7])?;
    model.kap_config.kap_co_option = kap_co_option_from_u8(cfg_tags[8])?;
    model.kap_config.kap_low_t_option = kap_low_t_option_from_u8(cfg_tags[9])?;
    model.kap_config.cubic_interpolation_in_x = cfg_tags[10] != 0;
    model.kap_config.cubic_interpolation_in_z = cfg_tags[11] != 0;
    model.kap_config.include_electron_conduction = cfg_tags[12] != 0;
    model.kap_config.use_blouin_conductive_opacities = cfg_tags[13] != 0;
    model.kap_config.use_zbase_for_type1 = cfg_tags[14] != 0;
    model.kap_config.use_type2_opacities = cfg_tags[15] != 0;

    let cfg_f64 = read_f64_vec(&mut pos, 10)?;
    model.eos_config.helm_blend_width = cfg_f64[0];
    model.kap_config.zbase = cfg_f64[1];
    model.kap_config.kap_blend_log_t_lower_bdy = cfg_f64[2];
    model.kap_config.kap_blend_log_t_upper_bdy = cfg_f64[3];
    model.kap_config.kap_type2_full_on_x = cfg_f64[4];
    model.kap_config.kap_type2_full_off_x = cfg_f64[5];
    model.kap_config.kap_type2_full_off_dz = cfg_f64[6];
    model.kap_config.kap_type2_full_on_dz = cfg_f64[7];
    model.kap_config.log_t_compton_blend_hi = cfg_f64[8];
    model.kap_config.log_r_compton_blend_lo = cfg_f64[9];

    expect_part(&mut pos, 6)?;
    let has_rsp = *bytes.get(pos).ok_or_else(|| eof("EOF has_rsp"))? != 0;
    pos += 1;
    let (saved_rsp, saved_rsp_model_vecs) = if has_rsp {
        let mut rsp = crate::rsp_def::RspState::new(nz);
        for dst in rsp.zone_vecs_mut() {
            *dst = read_f64_vec(&mut pos, nz)?;
        }
        let sc = read_f64_vec(&mut pos, 155)?;
        rsp.unpack_scalars(&sc);
        let mvecs = if model.controls.rsp_flag {
            let mut arr: [Vec<f64>; 10] = Default::default();
            for v in &mut arr {
                *v = read_f64_vec(&mut pos, nz)?;
            }
            Some(arr)
        } else {
            None
        };
        (Some(rsp), mvecs)
    } else {
        (None, None)
    };
    expect_part(&mut pos, 7)?;

    model.xmstar = (model.mstar - model.m_center).max(0.0);
    model.mstar_start = saved_mstar_start;
    if saved_lnd_start.len() == nz {
        model.lnd_start.clone_from(&saved_lnd_start);
        model.ln_t_start.clone_from(&saved_ln_t_start);
        model.ln_r_start.clone_from(&saved_ln_r_start);
        model.lum_start.clone_from(&saved_lum_start);
        model.v_start.clone_from(&saved_v_start);
        model.u_start.clone_from(&saved_u_start);
    } else {
        model.lnd_start.clone_from(&model.lnd);
        model.ln_t_start.clone_from(&model.ln_t);
        model.ln_r_start.clone_from(&model.ln_r);
        model.lum_start.clone_from(&model.lum);
        model.v_start.clone_from(&model.v);
        model.u_start.clone_from(&model.u);
    }
    model.sig.clone_from(&saved_sig);
    model.mlt_vc.clone_from(&saved_mlt_vc);
    model.aux.mlt_vc_old.clone_from(&saved_mlt_vc_old);
    model.set_qs();
    model.set_xqs();
    model.set_m_and_dm();
    model.set_m_grav_and_grav();
    model.pack_xh();
    if model.controls.rsp_flag {
        crate::micro::set_basic_vars(&mut model)?;
        if let Some([t, pg, pr, kap, cs, cp, r, m, dm, dmb]) =
            saved_rsp_model_vecs
        {
            (model.t, model.pgas, model.prad) = (t, pg, pr);
            (model.opacity, model.csound, model.cp) = (kap, cs, cp);
            (model.r, model.m, model.dm, model.dm_bar) = (r, m, dm, dmb);
            model.set_m_grav_and_grav();
            model.pack_xh();
            for k in 0..nz {
                model.rho[k] = model.lnd[k].exp();
                model.peos[k] = model.pgas[k] + model.prad[k];
                let r_in = if k + 1 < nz {
                    model.r[k + 1]
                } else {
                    model.r_center
                };
                model.rmid[k] = 0.5 * (model.r[k] + r_in);
            }
        } else {
            crate::micro::do_eos(&mut model)?;
            crate::micro::do_kap(&mut model)?;
        }
    } else {
        crate::hydro_vars::set_hydro_vars(&mut model, false, false)?;
        let dt_photo = model.dt;
        crate::struct_burn_mix::refresh_mixing_state(&mut model, dt_photo, &[]);
        model.save_start_values();
    }
    model.mstar_start = saved_mstar_start;
    if saved_lnd_start.len() == nz {
        model.lnd_start = saved_lnd_start;
        model.ln_t_start = saved_ln_t_start;
        model.ln_r_start = saved_ln_r_start;
        model.lum_start = saved_lum_start;
        model.v_start = saved_v_start;
        model.u_start = saved_u_start;
    }
    model.sig = saved_sig;
    model.mlt_vc = saved_mlt_vc;
    model.aux.mlt_vc_old = saved_mlt_vc_old;
    if let Some(rsp) = saved_rsp {
        model.aux.rsp = rsp;
    }

    Ok(model)
}
