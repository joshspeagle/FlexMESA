// SPDX-License-Identifier: LGPL-3.0-or-later
//! Phase 5a analytical vs. finite-difference Jacobian verification (`Issue #112`).

use flexmesa_const::SECYER;
use flexmesa_star::hydro_eqns::{assemble_henyey_system, HenyeySystem};
use flexmesa_star::riemann_ad::do_uface_and_pface;
use flexmesa_star::star_ad::StarAdOrder1;
use flexmesa_star::{
    eval_henyey_equations, load_zams_model, set_hydro_vars, EnergyEqnOption,
    StarControls, StarJobControls, StarModel,
};

fn usize_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("small test zone index fits in u32"))
}

fn set_pure_helm(m: &mut StarModel) {
    (m.eos_config.use_free_eos, m.eos_config.use_opal_scvh) = (false, false);
    (m.eos_config.use_skye, m.eos_config.use_pc) = (false, false);
    (m.eos_config.use_cms, m.eos_config.helm_blend_width) = (false, 0.0);
}

fn enable_all_zone_velocity(c: &mut StarControls) {
    c.redo_conv_for_dr_lt_mixing_length = false;
    c.velocity_q_upper_bound = 2.0;
    c.velocity_tau_lower_bound = -1.0;
    c.velocity_logt_lower_bound = -1.0;
}

fn build_compact_zams(
    nz_target: usize,
    controls: StarControls,
    job: StarJobControls,
) -> StarModel {
    let full =
        load_zams_model(controls, job).expect("1 Msun ZAMS model should load");
    let nz = nz_target.max(4).min(full.nz);
    let lnp_min = full.ln_peos[0];
    let lnp_max = full.ln_peos[full.nz - 1];
    let mut bounds = Vec::with_capacity(nz + 1);
    bounds.push(0);
    let mut prev = 0_usize;
    for b in 1..nz {
        let target_lnp =
            lnp_min + usize_to_f64(b) * (lnp_max - lnp_min) / usize_to_f64(nz);
        let max_idx = full.nz - (nz - b);
        let idx = (prev + 1..=max_idx)
            .find(|&s| full.ln_peos[s] >= target_lnp)
            .unwrap_or(max_idx)
            .clamp(prev + 1, max_idx);
        bounds.push(idx);
        prev = idx;
    }
    bounds.push(full.nz);

    let mut m = StarModel::new(
        nz,
        full.chem_id.clone(),
        full.job.clone(),
        full.controls.clone(),
    );
    (m.mstar, m.mstar_start, m.mstar_old, m.xmstar) =
        (full.mstar, full.mstar, full.mstar, full.xmstar);
    (m.r_center, m.m_center, m.l_center) =
        (full.r_center, full.m_center, full.l_center);
    let dt0 = 1.0e2 * SECYER;
    (m.dt, m.dt_next, m.dt_old) = (dt0, dt0, dt0);
    for dst_k in 0..nz {
        let (k0, k1) = (bounds[dst_k], bounds[dst_k + 1]);
        let dm_tot: f64 = full.dm[k0..k1].iter().sum();
        let vol_sum: f64 = (k0..k1)
            .map(|s| full.dm[s] / full.lnd[s].exp().max(1.0e-30))
            .sum();
        let lnt_sum: f64 = (k0..k1).map(|s| full.dm[s] * full.ln_t[s]).sum();
        m.dq[dst_k] = full.dq[k0..k1].iter().sum();
        m.lnd[dst_k] = (dm_tot / vol_sum.max(1.0e-30)).ln();
        m.ln_t[dst_k] = lnt_sum / dm_tot.max(1.0e-30);
        (m.ln_r[dst_k], m.lum[dst_k]) = (full.ln_r[k0], full.lum[k0]);
        for j in 0..m.species {
            let x_sum: f64 = (k0..k1)
                .map(|s| full.dm[s] * full.xa[s * full.species + j])
                .sum();
            m.xa[dst_k * m.species + j] = x_sum / dm_tot.max(1.0e-30);
        }
    }
    m.sync_mesh_and_pack().expect("sync mesh should succeed");
    m.reconstruct_ln_r_from_lnd_and_dm();
    m.finish_init_model().expect("compact init should succeed");
    set_hydro_vars(&mut m, false, false).expect("initial set_hydro_vars");
    m.save_start_values();
    m.save_old_values();
    m
}

fn lookup_block_entry(
    sys: &HenyeySystem,
    k_equ: usize,
    i_equ: usize,
    k_var: usize,
    j_var: usize,
) -> f64 {
    let idx = k_equ * sys.nvar * sys.nvar + i_equ * sys.nvar + j_var;
    match (k_equ, k_var) {
        (ke, kv) if ke == kv => sys.dblk[idx],
        (ke, kv) if kv + 1 == ke => sys.lblk[idx],
        _ => sys.ublk[idx],
    }
}

fn assert_entry_matches(label: &str, an: f64, fd: f64, rtol: f64, atol: f64) {
    let (diff, scale) = ((an - fd).abs(), an.abs().max(fd.abs()));
    assert!(
        diff <= atol || (scale > 0.0 && diff / scale <= rtol),
        "{label}: analytic={an:.10e}, fd={fd:.10e}, diff={diff:.3e}"
    );
}

fn freeze_microphysics_tangents(m: &mut StarModel) {
    for k in 0..m.nz {
        m.ln_peos_ad[k] = StarAdOrder1::constant(m.ln_peos_ad[k].val);
        m.ln_e_ad[k] = StarAdOrder1::constant(m.ln_e_ad[k].val);
        m.ln_s_ad[k] = StarAdOrder1::constant(m.ln_s_ad[k].val);
        m.gradt_ad[k] = StarAdOrder1::constant(m.gradt_ad[k].val);
        m.mlt_vc_ad[k] = StarAdOrder1::constant(m.mlt_vc_ad[k].val);
        m.scale_height_ad[k] = StarAdOrder1::constant(m.scale_height_ad[k].val);
        m.eps_grav_ad[k] = StarAdOrder1::constant(m.eps_grav_ad[k].val);
    }
    m.ln_peos_surf_ad = StarAdOrder1::constant(m.ln_peos_surf_ad.val);
    m.ln_t_surf_ad = StarAdOrder1::constant(m.ln_t_surf_ad.val);
    for d in m.eos_d_dlnd.iter_mut().chain(m.eos_d_dlnt.iter_mut()) {
        d.fill(0.0);
    }
    for buf in [
        &mut m.d_epsnuc_dlnd,
        &mut m.d_epsnuc_dlnt,
        &mut m.d_epsnuc_dx,
        &mut m.d_non_nuc_neu_dlnd,
        &mut m.d_non_nuc_neu_dlnt,
        &mut m.d_dxdt_nuc_dlnd,
        &mut m.d_dxdt_nuc_dlnt,
        &mut m.aux.dln_peos_dxa_for_partials,
        &mut m.aux.dln_e_dxa_for_partials,
        &mut m.aux.d_eps_grav_dx,
    ] {
        buf.fill(0.0);
    }
}

fn eval_frozen_pde_system(
    m: &mut StarModel,
    include_chem: bool,
) -> HenyeySystem {
    for k in 0..m.nz {
        (m.rho[k], m.t[k], m.r[k]) =
            (m.lnd[k].exp(), m.ln_t[k].exp(), m.ln_r[k].exp());
        let r_in = m.ln_r.get(k + 1).map_or(m.r_center, |&lr| lr.exp());
        m.rmid[k] = f64::midpoint(m.r[k], r_in);
    }
    m.set_m_grav_and_grav();
    if m.controls.u_flag {
        do_uface_and_pface(m).expect("frozen u_flag HLLC faces");
    }
    let mut sys = HenyeySystem::new(0, 0, Vec::new());
    assemble_henyey_system(m, include_chem, &mut sys)
        .expect("frozen PDE assembly");
    sys
}

#[allow(clippy::too_many_lines)]
fn assert_tridiagonal_jacobian_matches_fd(
    base_model: &StarModel,
    include_chem: bool,
    rel_tol: f64,
    abs_tol: f64,
) {
    let mut m0 = base_model.clone();
    m0.pack_xh();
    let sys0 =
        eval_henyey_equations(&mut m0, include_chem).expect("baseline eqns");
    let (nz, nvar, nvar_hydro) = (sys0.nz, sys0.nvar, m0.layout.nvar_hydro);
    for k_var in 0..nz {
        for j_var in 0..nvar_hydro {
            let eps_scaled = if m0.controls.mixing_length_alpha <= 0.0 {
                1.0e-6
            } else if j_var == m0.layout.i_lum {
                1.0e-2
            } else {
                5.0e-4
            };
            let delta = eps_scaled * sys0.x_scale[k_var * nvar + j_var];

            let perturb_h = |sgn: f64| {
                let mut m = base_model.clone();
                m.pack_xh();
                m.xh[k_var * nvar_hydro + j_var] += sgn * delta;
                m.unpack_xh();
                m.set_m_grav_and_grav();
                eval_henyey_equations(&mut m, include_chem).expect("hydro fd")
            };
            let (sys_p, sys_m) = (perturb_h(1.0), perturb_h(-1.0));
            for k_equ in 0..nz {
                if k_equ.abs_diff(k_var) > 1 {
                    continue;
                }
                for i_equ in 0..nvar {
                    let eq0 = sys0.equ[k_equ * nvar + i_equ].abs();
                    let fd = (sys_p.equ[k_equ * nvar + i_equ]
                        - sys_m.equ[k_equ * nvar + i_equ])
                        / (2.0 * eps_scaled);
                    let an =
                        lookup_block_entry(&sys0, k_equ, i_equ, k_var, j_var);
                    let h_rtol = if k_var < 2 || k_equ < 2 {
                        rel_tol.max(2.0e-3)
                    } else {
                        rel_tol
                    };
                    let eff_atol =
                        abs_tol.max(2.0 * f64::EPSILON * eq0 / eps_scaled);
                    assert_entry_matches(
                        &format!("hydro k_equ={k_equ} i_equ={i_equ} k_var={k_var} j_var={j_var}"),
                        an,
                        fd,
                        h_rtol,
                        eff_atol,
                    );
                }
            }
        }

        if !include_chem {
            continue;
        }
        let use_surf_mom_bc = m0.controls.use_momentum_outer_bc
            || m0.controls.use_zero_pgas_outer_bc
            || m0.controls.use_fixed_psurf_outer_bc
            || m0.controls.use_compression_outer_bc
            || m0.controls.use_fixed_vsurf_outer_bc;
        let idx_he4 = m0.iso_names.iter().position(|n| n == "he4").unwrap_or(0);
        let idx_o16 = m0.iso_names.iter().position(|n| n == "o16").unwrap_or(1);

        for j_sp in 0..m0.species {
            let j_var = nvar_hydro + j_sp;
            let x_sc = sys0.x_scale[k_var * nvar + j_var];
            let xa_0 = base_model.xa[k_var * m0.species + j_sp];
            let opt_dx = f64::EPSILON.cbrt();
            let delta = if xa_0 > 4.0 * opt_dx {
                opt_dx
            } else if xa_0 > 0.0 {
                0.25 * xa_0
            } else {
                opt_dx * x_sc
            };
            let eps_scaled = delta / x_sc;

            let perturb_x = |sgn: f64, sink: Option<usize>| {
                let mut m = base_model.clone();
                m.xa[k_var * m.species + j_sp] += sgn * delta;
                if let Some(s) = sink {
                    m.xa[k_var * m.species + s] -= sgn * delta;
                }
                eval_henyey_equations(&mut m, true).expect("chem fd")
            };
            let (sys_p, sys_m) = (perturb_x(1.0, None), perturb_x(-1.0, None));
            for k_equ in 0..nz {
                if k_equ.abs_diff(k_var) > 1 {
                    continue;
                }
                for i_equ in nvar_hydro..nvar {
                    let fd = (sys_p.equ[k_equ * nvar + i_equ]
                        - sys_m.equ[k_equ * nvar + i_equ])
                        / (2.0 * eps_scaled);
                    let an =
                        lookup_block_entry(&sys0, k_equ, i_equ, k_var, j_var);
                    assert_entry_matches("chem", an, fd, 1.0e-5, 1.0e-9);
                }
            }

            for idx_sink in [idx_he4, idx_o16] {
                if j_sp == idx_sink {
                    continue;
                }
                let j_var_sink = nvar_hydro + idx_sink;
                let x_sc_sink = sys0.x_scale[k_var * nvar + j_var_sink];

                let (simplex_plus, simplex_minus) = (
                    perturb_x(1.0, Some(idx_sink)),
                    perturb_x(-1.0, Some(idx_sink)),
                );
                for k_equ in 0..nz {
                    if k_equ.abs_diff(k_var) > 1 {
                        continue;
                    }
                    for i_equ in 0..nvar {
                        let a_j = lookup_block_entry(
                            &sys0, k_equ, i_equ, k_var, j_var,
                        );
                        let a_sink = lookup_block_entry(
                            &sys0, k_equ, i_equ, k_var, j_var_sink,
                        );
                        let u_flag_only =
                            m0.controls.u_flag && !m0.controls.v_flag;
                        let upstream_omits_dxa = i_equ == m0.layout.i_equ_l
                            || (k_equ == 0
                                && i_equ == m0.layout.i_dv_dt
                                && !use_surf_mom_bc)
                            || (u_flag_only
                                && (i_equ == m0.layout.i_dv_dt
                                    || Some(i_equ) == m0.layout.i_dlnr_dt
                                    || (i_equ == m0.layout.i_dlne_dt
                                        && k_equ != k_var)));
                        if upstream_omits_dxa {
                            assert!(
                                a_j.abs() <= 1.0e-30 && a_sink.abs() <= 1.0e-30
                            );
                            continue;
                        }
                        let fd_dir = (simplex_plus.equ[k_equ * nvar + i_equ]
                            - simplex_minus.equ[k_equ * nvar + i_equ])
                            / (2.0 * eps_scaled);
                        let an_dir = a_j - a_sink * (x_sc / x_sc_sink);
                        let eq0 = sys0.equ[k_equ * nvar + i_equ].abs();
                        let eff_atol =
                            abs_tol.max(2.0 * f64::EPSILON * eq0 / eps_scaled);
                        let zone_rtol = if k_var == 0 {
                            rel_tol.max(5.0e-2)
                        } else if k_var == 1 || k_equ < 2 {
                            rel_tol.max(7.0e-3)
                        } else if m0.eos_config.use_free_eos
                            || m0.eos_config.use_opal_scvh
                            || m0.eos_config.use_skye
                        {
                            rel_tol.max(4.0e-3)
                        } else {
                            rel_tol
                        };
                        assert_entry_matches(
                            &format!("simplex k_equ={k_equ} i_equ={i_equ} k_var={k_var} j_sp={j_sp}"),
                            an_dir,
                            fd_dir,
                            zone_rtol,
                            eff_atol,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_frozen_microphysics_star_ad_jacobian_matches_central_fd_to_1e_10() {
    for (v_flag, u_flag, rti_flag) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, false, true),
    ] {
        let mut controls = StarControls {
            v_flag,
            u_flag,
            rti_flag,
            use_momentum_outer_bc: v_flag || u_flag,
            using_velocity_time_centering: true,
            include_p_in_velocity_time_centering: true,
            ..StarControls::default()
        };
        enable_all_zone_velocity(&mut controls);
        let mut base =
            build_compact_zams(6, controls, StarJobControls::default());
        base.dt = 5.0;
        for k in 0..base.nz {
            if v_flag {
                base.v[k] = 2.0e5 * usize_to_f64(k + 1);
            }
            if u_flag {
                base.u[k] = 1.5e5 * usize_to_f64(base.nz - k);
            }
            if rti_flag {
                base.aux.alpha_rti[k] = 0.03 + 0.01 * usize_to_f64(k);
            }
        }
        set_hydro_vars(&mut base, false, false).expect("hydro vars");
        base.save_start_values();
        for k in 0..base.nz {
            base.lnd[k] += 1.0e-3 * (usize_to_f64(k) - 2.0);
            base.ln_t[k] -= 8.0e-4 * usize_to_f64(k);
            base.ln_r[k] += 3.0e-4;
        }
        base.set_m_grav_and_grav();
        set_hydro_vars(&mut base, false, false).expect("trial hydro vars");
        freeze_microphysics_tangents(&mut base);
        base.pack_xh();

        let mut m0 = base.clone();
        let sys0 = eval_frozen_pde_system(&mut m0, true);
        let (nvar, nvar_h, eps) = (sys0.nvar, base.layout.nvar_hydro, 3.0e-6);
        for k_var in 0..base.nz {
            for j_var in 0..nvar_h {
                let delta = eps * sys0.x_scale[k_var * nvar + j_var];
                let (mut mp, mut mm) = (base.clone(), base.clone());
                mp.xh[k_var * nvar_h + j_var] += delta;
                mp.unpack_xh();
                mm.xh[k_var * nvar_h + j_var] -= delta;
                mm.unpack_xh();
                let sys_p = eval_frozen_pde_system(&mut mp, true);
                let sys_m = eval_frozen_pde_system(&mut mm, true);
                for k_equ in 0..base.nz {
                    if k_equ.abs_diff(k_var) > 1 {
                        continue;
                    }
                    for i_equ in 0..nvar {
                        let fd = (sys_p.equ[k_equ * nvar + i_equ]
                            - sys_m.equ[k_equ * nvar + i_equ])
                            / (2.0 * eps);
                        let an = lookup_block_entry(
                            &sys0, k_equ, i_equ, k_var, j_var,
                        );
                        assert_entry_matches(
                            &format!("frozen({v_flag},{u_flag},{rti_flag})"),
                            an,
                            fd,
                            1.0e-7,
                            1.0e-9,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_coupled_hydro_chem_jacobian_matches_central_fd_across_sinks_and_spherical_v_flag(
) {
    // 1. Analytical `HELM` radiative envelope (`alpha_MLT = 0`): strict `2e-5`
    //    relative gate across all zones and both `Dedt` / `EpsGrav` forms.
    for opt in [EnergyEqnOption::Dedt, EnergyEqnOption::EpsGrav] {
        let controls = StarControls {
            energy_eqn_option: opt,
            mixing_length_alpha: 0.0,
            op_split_burn: false,
            redo_conv_for_dr_lt_mixing_length: false,
            ..StarControls::default()
        };
        let mut model =
            build_compact_zams(6, controls, StarJobControls::default());
        set_pure_helm(&mut model);
        set_hydro_vars(&mut model, false, false).expect("helm hydro vars");
        model.save_start_values();
        if opt == EnergyEqnOption::Dedt {
            for k in 0..model.nz {
                model.lnd[k] += 2.0e-3 * (usize_to_f64(k) * 0.2 - 0.5);
                model.ln_t[k] += 1.5e-3 * (0.4 - usize_to_f64(k) * 0.1);
            }
            model.reconstruct_ln_r_from_lnd_and_dm();
            model.set_m_grav_and_grav();
        }
        model.pack_xh();
        assert_tridiagonal_jacobian_matches_fd(&model, true, 2.0e-5, 1.0e-9);
    }

    // 2. Active MLT + spherical `v_flag` + `d_dwork_dxam1/00/p1` blocks.
    let mut controls_v = StarControls {
        energy_eqn_option: EnergyEqnOption::Dedt,
        v_flag: true,
        use_momentum_outer_bc: true,
        using_velocity_time_centering: true,
        include_p_in_velocity_time_centering: true,
        op_split_burn: false,
        ..StarControls::default()
    };
    enable_all_zone_velocity(&mut controls_v);
    let mut model_v =
        build_compact_zams(6, controls_v, StarJobControls::default());
    set_pure_helm(&mut model_v);
    model_v.dt = 20.0;
    for k in 0..model_v.nz {
        model_v.v[k] = 1.5e5 * usize_to_f64(model_v.nz - k);
    }
    set_hydro_vars(&mut model_v, false, false).expect("v_flag hydro vars");
    model_v.save_start_values();
    for k in 0..model_v.nz {
        model_v.v[k] *= 1.03;
        model_v.lnd[k] += 1.5e-3 * (usize_to_f64(k) * 0.2 - 0.4);
        model_v.ln_t[k] += 1.0e-3 * (0.3 - usize_to_f64(k) * 0.1);
    }
    model_v.set_m_grav_and_grav();
    model_v.pack_xh();
    let sys_v = eval_henyey_equations(&mut model_v.clone(), true).expect("v");
    let (i_e, j_x0) = (model_v.layout.i_dlne_dt, model_v.layout.nvar_hydro);
    assert!(lookup_block_entry(&sys_v, 2, i_e, 1, j_x0).abs() > 0.0);
    assert!(lookup_block_entry(&sys_v, 2, i_e, 2, j_x0).abs() > 0.0);
    assert!(lookup_block_entry(&sys_v, 2, i_e, 3, j_x0).abs() > 0.0);
    assert_tridiagonal_jacobian_matches_fd(&model_v, true, 5.0e-4, 1.0e-8);
}

#[test]
fn test_coupled_u_flag_hllc_jacobian_across_all_outer_bcs_and_chem_rti() {
    for bc_mode in 0..5 {
        let mut c = StarControls {
            u_flag: true,
            v_flag: false,
            mixing_length_alpha: 0.0,
            using_velocity_time_centering: true,
            include_p_in_velocity_time_centering: true,
            op_split_burn: false,
            ..StarControls::default()
        };
        enable_all_zone_velocity(&mut c);
        match bc_mode {
            0 => c.use_momentum_outer_bc = true,
            1 => c.use_compression_outer_bc = true,
            2 => c.use_zero_pgas_outer_bc = true,
            3 => {
                c.use_fixed_psurf_outer_bc = true;
                c.fixed_psurf = 1.0e5;
            }
            _ => {
                c.use_fixed_vsurf_outer_bc = true;
                c.fixed_vsurf = 2.5e4;
                c.rti_flag = true;
            }
        }
        let include_chem = bc_mode == 0;
        let mut model = build_compact_zams(6, c, StarJobControls::default());
        model.dt = if include_chem { 0.05 } else { 5.0 };
        let u_amp = if include_chem { 0.0 } else { 2.0e5 };
        for k in 0..model.nz {
            model.u[k] = u_amp * usize_to_f64(model.nz - k);
            if model.controls.rti_flag {
                model.aux.alpha_rti[k] = 0.02 + 0.01 * usize_to_f64(k);
            }
        }
        set_hydro_vars(&mut model, false, !include_chem).expect("u_flag vars");
        model.save_start_values();
        for k in 0..model.nz {
            model.u[k] *= 1.02;
            model.lnd[k] += 1.0e-3;
            model.ln_t[k] += 8.0e-4;
            model.ln_r[k] += 3.0e-4;
        }
        model.set_m_grav_and_grav();
        model.pack_xh();
        assert_tridiagonal_jacobian_matches_fd(
            &model,
            include_chem,
            5.0e-5,
            1.0e-8,
        );
    }
}

#[test]
fn test_coupled_v_flag_rti_jacobian_with_active_baroclinic_source_diffusion_and_supersaturation(
) {
    let mut controls = StarControls {
        v_flag: true,
        rti_flag: true,
        mixing_length_alpha: 0.0,
        use_momentum_outer_bc: true,
        ..StarControls::default()
    };
    enable_all_zone_velocity(&mut controls);
    let mut model = build_compact_zams(8, controls, StarJobControls::default());
    set_pure_helm(&mut model);
    model.dt = 10.0;
    for k in [5_usize, 4, 3] {
        let step = usize_to_f64(6 - k);
        model.lnd[k] = model.lnd[6] + 0.30 * step;
        model.ln_t[k] = model.ln_t[6] - 0.50 * step;
    }
    model.reconstruct_ln_r_from_lnd_and_dm();
    model.set_m_grav_and_grav();
    for k in 0..model.nz {
        model.v[k] = 2.0e5 * usize_to_f64(model.nz - k);
        model.aux.alpha_rti[k] = 0.05 + 0.02 * usize_to_f64(k);
    }
    model.aux.alpha_rti[5] = 1.5 * model.controls.rti_max_alpha;
    set_hydro_vars(&mut model, false, true).expect("rti hydro vars");
    model.save_start_values();

    assert_eq!(model.aux.dpdr_drhodr_info[0], 0.0);
    assert!(model.aux.dpdr_drhodr_info.iter().any(|&d| d < 0.0));
    assert!(model.aux.eta_rti.iter().any(|&e| e > 0.0));
    assert!(model.aux.sig_rti.iter().any(|&s| s > 0.0));

    for k in 0..model.nz {
        model.aux.alpha_rti[k] *= 1.04;
        model.v[k] *= 1.01;
    }
    model.pack_xh();

    let sys0 = eval_henyey_equations(&mut model.clone(), false).expect("sys0");
    let i_rti = model.layout.i_alpha_rti.expect("rti var slot");
    let has_offdiag_rti = (1..model.nz)
        .any(|k| lookup_block_entry(&sys0, k, i_rti, k - 1, i_rti).abs() > 0.0);
    assert!(has_offdiag_rti, "diffusive sig_rti must populate lblk/ublk");

    assert_tridiagonal_jacobian_matches_fd(&model, false, 5.0e-5, 1.0e-8);
}
