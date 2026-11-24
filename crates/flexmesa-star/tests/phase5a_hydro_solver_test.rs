// SPDX-License-Identifier: GPL-3.0-or-later
//! Phase 5a analytical vs. finite-difference Jacobian verification (`Issue #112`).

use flexmesa_const::SECYER;
use flexmesa_star::hydro_eqns::{HenyeySystem, assemble_henyey_system};
use flexmesa_star::riemann_ad::do_uface_and_pface;
use flexmesa_star::star_ad::StarAdOrder1;
use flexmesa_star::{
    EnergyEqnOption, StarControls, StarJobControls, StarModel,
    eval_henyey_equations, load_zams_model, set_hydro_vars,
};

/// Zone counts of the compact fixture the Jacobian comparisons run on. The
/// 12- and 24-zone meshes reach regimes the 6-zone mesh does not (#307).
const FIXTURE_ZONES: [usize; 3] = [6, 12, 24];

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
    mut controls: StarControls,
    job: StarJobControls,
) -> StarModel {
    if controls.zbase < 0.0 {
        controls.zbase = 0.02;
    }
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
            .map(|s| full.dm[s] / flexmesa_math::exp(full.lnd[s]).max(1.0e-30))
            .sum();
        let lnt_sum: f64 = (k0..k1).map(|s| full.dm[s] * full.ln_t[s]).sum();
        m.dq[dst_k] = full.dq[k0..k1].iter().sum();
        m.lnd[dst_k] = flexmesa_math::log(dm_tot / vol_sum.max(1.0e-30));
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
    m.save_start_values().expect("save_start_values");
    flexmesa_star::record_start_of_step_energy_integrals(&mut m);
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

/// Bound on the error of a central difference of the temperature-gradient
/// residual at face `k` caused by rounding inside `calc_MLT`.
///
/// Upstream evaluates the convective efficiency factor as
/// `Zeta = exp(3d0*log(Gamma) - log(Bcubed))` (`turb/private/mlt.f90:164`) and
/// then `gradT = (1d0 - Zeta)*gradr + Zeta*gradL` (`mlt.f90:183`). For very
/// efficient convection `Zeta` is within `1/Gamma` of one, so the exponent is a
/// difference of two numbers near `3 ln(Gamma)` whose rounding error is a few
/// units in the last place of `3 ln(Gamma)`. That error reaches `gradT`
/// multiplied by `gradr - gradL`, and the residual `gradT*dlnP - dlnT`
/// (`star/private/hydro_temperature.f90:55-75`) multiplied by `dlnP`. The
/// analytic derivative does not carry this error; the finite difference does,
/// once in each of its two evaluations. On the 12-zone fixture at face 5
/// (`Gamma = 8.5e8`, `gradr - gradL = 2.3e5`) central differences of `gradT`
/// alone scatter by 0.01 to 0.9 per cent as the step is varied (#307).
fn mlt_zeta_fd_noise(
    m0: &StarModel,
    k: usize,
    i_equ: usize,
    eps_scaled: f64,
) -> f64 {
    if i_equ != m0.layout.i_equ_l || k == 0 || m0.mlt_gamma[k] <= 1.0 {
        return 0.0;
    }
    let p_cell = flexmesa_star::eval_cell_peos_time_centered_ad(m0, k, false).0.val;
    let p_above =
        flexmesa_star::eval_cell_peos_time_centered_ad(m0, k - 1, true).0.val;
    let dln_p = (p_above - p_cell).abs() / p_cell.min(p_above);
    // Four units in the last place of the exponent for each evaluation: one
    // for each logarithm, and two for the error `Gamma` and `Bcubed` already
    // carry, which the factor of three multiplies.
    let exponent_error =
        4.0 * f64::EPSILON * 3.0 * flexmesa_math::log(m0.mlt_gamma[k]);
    let grad_t_error = exponent_error * (m0.gradr[k] - m0.gradl[k]).abs();
    2.0 * grad_t_error * dln_p / (2.0 * eps_scaled)
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
    m.surf_pt = flexmesa_star::SurfPt {
        ln_t: m.surf_pt.ln_t,
        ln_p: m.surf_pt.ln_p,
        ..flexmesa_star::SurfPt::default()
    };
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
        (m.rho[k], m.t[k], m.r[k]) = (
            flexmesa_math::exp(m.lnd[k]),
            flexmesa_math::exp(m.ln_t[k]),
            flexmesa_math::exp(m.ln_r[k]),
        );
        let r_in = m
            .ln_r
            .get(k + 1)
            .map_or(m.r_center, |&lr| flexmesa_math::exp(lr));
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
    // Upstream Newton residual/Jacobian evaluation (`set_vars_for_solver`,
    // `hydro_vars.f90:275`) runs with `skip_mixing_info = .true.` so
    // start-of-step `sig`, `sig_rti`, `sigmid_rti`, and `dpdr_drhodr_info` stay
    // fixed across solver iterations.
    m0.aux.doing_struct_burn_mix = true;
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
                m.aux.doing_struct_burn_mix = true;
                m.pack_xh();
                m.xh[k_var * nvar_hydro + j_var] += sgn * delta;
                m.unpack_xh();
                m.set_m_grav_and_grav();
                eval_henyey_equations(&mut m, include_chem).expect("hydro fd")
            };
            let (sys_p, sys_m) = (perturb_h(1.0), perturb_h(-1.0));
            // A second central difference at half the step measures how far
            // the finite difference itself can be trusted for this entry.
            let (half_plus, half_minus) = (perturb_h(0.5), perturb_h(-0.5));
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
                    // The two step sizes differ by the truncation error
                    // the full step carries beyond the half step (3/4 of
                    // it for a smooth residual) plus any evaluation noise,
                    // so twice their difference bounds the error of `fd`.
                    // The analytic entry cannot be checked more tightly
                    // than the reference it is compared with.
                    let fd_half = (half_plus.equ[k_equ * nvar + i_equ]
                        - half_minus.equ[k_equ * nvar + i_equ])
                        / eps_scaled;
                    let fd_uncertainty = 2.0 * (fd - fd_half).abs();
                    let eff_atol = abs_tol
                        .max(2.0 * f64::EPSILON * eq0 / eps_scaled)
                        + fd_uncertainty
                        + mlt_zeta_fd_noise(&m0, k_equ, i_equ, eps_scaled);
                    assert_entry_matches(
                        &format!(
                            "hydro k_equ={k_equ} i_equ={i_equ} k_var={k_var} j_var={j_var}"
                        ),
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
            let opt_dx = flexmesa_math::pow(f64::EPSILON, 1.0 / 3.0);
            let delta = if xa_0 > 4.0 * opt_dx {
                opt_dx
            } else if xa_0 > 0.0 {
                0.25 * xa_0
            } else {
                opt_dx * x_sc
            };
            let eps_scaled = delta / x_sc;

            let perturb_x_with_p_surf = |sgn: f64, sink: Option<usize>| {
                let mut m = base_model.clone();
                m.aux.doing_struct_burn_mix = true;
                m.xa[k_var * m.species + j_sp] += sgn * delta;
                if let Some(s) = sink {
                    m.xa[k_var * m.species + s] -= sgn * delta;
                }
                let sys = eval_henyey_equations(&mut m, true).expect("chem fd");
                let p_surf = flexmesa_math::exp(m.ln_peos_surf_ad.val);
                (sys, p_surf, m.energy)
            };
            let perturb_x =
                |sgn: f64, sink: Option<usize>| perturb_x_with_p_surf(sgn, sink).0;
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

                let (
                    (simplex_plus, simplex_plus_p_surf, energy_plus),
                    (simplex_minus, simplex_minus_p_surf, energy_minus),
                ) = (
                    perturb_x_with_p_surf(1.0, Some(idx_sink)),
                    perturb_x_with_p_surf(-1.0, Some(idx_sink)),
                );
                // Divisor of the surface momentum residual
                // (`hydro_momentum.rs::do1_momentum_eqn`, `k == 0`).
                let p_cell_0 =
                    flexmesa_star::eval_cell_peos_time_centered_ad(&m0, 0, false)
                        .0
                        .val;
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
                        let mut an_dir = a_j - a_sink * (x_sc / x_sc_sink);
                        if k_var == 0
                            && k_equ == 0
                            && i_equ == m0.layout.i_dv_dt
                            && use_surf_mom_bc
                        {
                            // Upstream's surface momentum equation takes the
                            // pressure above cell 1 from `P_surf_ad`
                            // (`star/private/hydro_momentum.f90:495-496`),
                            // which carries no composition partials, so the
                            // analytic block omits
                            // `d(residual)/d(P_surf) * d(P_surf)/d(xa)`. The
                            // finite difference includes it, because the
                            // atmosphere is re-evaluated with the perturbed
                            // surface opacity. Add the omitted term to the
                            // analytic side, from a central difference of
                            // `P_surf` alone, so the remaining entries are
                            // compared at full strength.
                            an_dir += -(simplex_plus_p_surf
                                - simplex_minus_p_surf)
                                / (2.0 * eps_scaled)
                                / p_cell_0;
                        }
                        let dedt_form = u_flag_only
                            || m0.controls.energy_eqn_option
                                == EnergyEqnOption::Dedt;
                        if dedt_form
                            && i_equ == m0.layout.i_dlne_dt
                            && k_equ == k_var
                            && m0.dt > 0.0
                        {
                            // The `dedt` form of the energy equation takes
                            // its composition partial from the equation of
                            // state, `-scal*(energy/dt)*dlnE_dxa_for_partials`
                            // (`star/private/hydro_energy.f90:491-496`). That
                            // partial does not follow the full composition
                            // dependence of the tabulated components: on the
                            // 12-zone fixture it differs from a central
                            // difference of the specific energy by up to 0.5
                            // per cent when the metal fraction changes, and
                            // that difference accounts for the whole mismatch
                            // in this entry to seven digits (#307). This test
                            // checks how the equation is assembled, so the
                            // difference between the two is moved to the
                            // analytic side here. The accuracy of the
                            // equation-of-state partial itself belongs to the
                            // equation-of-state tests.
                            let sp = m0.species;
                            let dlne =
                                |j: usize| m0.aux.dln_e_dxa_for_partials[k_equ * sp + j];
                            let an_de = m0.energy[k_equ]
                                * x_sc
                                * (dlne(j_sp) - dlne(idx_sink));
                            let fd_de = (energy_plus[k_equ] - energy_minus[k_equ])
                                / (2.0 * eps_scaled);
                            an_dir += -flexmesa_star::hydro_energy::set_energy_eqn_scal(
                                &m0, k_equ,
                            )
                            .expect("energy scal") / m0.dt
                                * (fd_de - an_de);
                        }
                        let eq0 = sys0.equ[k_equ * nvar + i_equ].abs();
                        let eff_atol =
                            abs_tol.max(2.0 * f64::EPSILON * eq0 / eps_scaled);
                        let zone_rtol = if k_var <= 1 || k_equ < 2 {
                            rel_tol.max(2.0e-3)
                        } else if m0.eos_config.use_free_eos
                            || m0.eos_config.use_opal_scvh
                            || m0.eos_config.use_skye
                        {
                            rel_tol.max(4.0e-3)
                        } else {
                            rel_tol
                        };
                        assert_entry_matches(
                            &format!(
                                "simplex k_equ={k_equ} i_equ={i_equ} k_var={k_var} j_sp={j_sp}"
                            ),
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
    for nz in FIXTURE_ZONES {
        frozen_microphysics_star_ad_jacobian_matches_central_fd_to_1e_10_at(nz);
    }
}

fn frozen_microphysics_star_ad_jacobian_matches_central_fd_to_1e_10_at(nz: usize) {
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
            build_compact_zams(nz, controls, StarJobControls::default());
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
        base.save_start_values().expect("save_start_values");
        flexmesa_star::record_start_of_step_energy_integrals(&mut base);
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
                let eps_var = if j_var == base.layout.i_lum
                    || Some(j_var) == base.layout.i_v
                {
                    1.0e-4
                } else {
                    eps
                };
                let delta = eps_var * sys0.x_scale[k_var * nvar + j_var];
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
                            / (2.0 * eps_var);
                        let an = lookup_block_entry(
                            &sys0, k_equ, i_equ, k_var, j_var,
                        );
                        let eq0 = sys0.equ[k_equ * nvar + i_equ].abs();
                        // Literal provenance (#155): pre-#262 used fixed atol
                        // `1.0e-9`; #262 added the IEEE-754 central-FD
                        // cancellation floor `2 * f64::EPSILON * |equ| / eps_var`
                        // (matching line 233) because on `u_flag` (`k_equ = 2`,
                        // `i_equ = 3`) `|equ| = 7.05e6` makes 1 ULP of central
                        // FD equal to `2.6e-4` at `eps_var = 3e-6`.
                        let eff_atol =
                            1.0e-9_f64.max(2.0 * f64::EPSILON * eq0 / eps_var);
                        assert_entry_matches(
                            &format!(
                                "frozen({v_flag},{u_flag},{rti_flag}) k_equ={k_equ} i_equ={i_equ} k_var={k_var} j_var={j_var}"
                            ),
                            an,
                            fd,
                            1.0e-7,
                            eff_atol,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_coupled_hydro_chem_jacobian_matches_central_fd_across_sinks_and_spherical_v_flag() {
    for nz in FIXTURE_ZONES {
        coupled_hydro_chem_jacobian_matches_central_fd_across_sinks_and_spherical_v_flag_at(nz);
    }
}

fn coupled_hydro_chem_jacobian_matches_central_fd_across_sinks_and_spherical_v_flag_at(nz: usize) {
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
            build_compact_zams(nz, controls, StarJobControls::default());
        set_pure_helm(&mut model);
        set_hydro_vars(&mut model, false, false).expect("helm hydro vars");
        model.save_start_values().expect("save_start_values");
        flexmesa_star::record_start_of_step_energy_integrals(&mut model);
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
        build_compact_zams(nz, controls_v, StarJobControls::default());
    set_pure_helm(&mut model_v);
    model_v.dt = 20.0;
    for k in 0..model_v.nz {
        model_v.v[k] = 1.5e5 * usize_to_f64(model_v.nz - k);
    }
    set_hydro_vars(&mut model_v, false, false).expect("v_flag hydro vars");
    model_v.save_start_values().expect("save_start_values");
    flexmesa_star::record_start_of_step_energy_integrals(&mut model_v);
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
    for nz in FIXTURE_ZONES {
        coupled_u_flag_hllc_jacobian_across_all_outer_bcs_and_chem_rti_at(nz);
    }
}

fn coupled_u_flag_hllc_jacobian_across_all_outer_bcs_and_chem_rti_at(nz: usize) {
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
        let mut model = build_compact_zams(nz, c, StarJobControls::default());
        model.dt = if include_chem { 0.05 } else { 5.0 };
        let u_amp = if include_chem { 0.0 } else { 2.0e5 };
        for k in 0..model.nz {
            model.u[k] = u_amp * usize_to_f64(model.nz - k);
            if model.controls.rti_flag {
                model.aux.alpha_rti[k] = 0.02 + 0.01 * usize_to_f64(k);
            }
        }
        set_hydro_vars(&mut model, false, !include_chem).expect("u_flag vars");
        model.save_start_values().expect("save_start_values");
        flexmesa_star::record_start_of_step_energy_integrals(&mut model);
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
fn test_coupled_v_flag_rti_jacobian_with_active_baroclinic_source_diffusion_and_supersaturation()
 {
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
    model.save_start_values().expect("save_start_values");
    flexmesa_star::record_start_of_step_energy_integrals(&mut model);

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
