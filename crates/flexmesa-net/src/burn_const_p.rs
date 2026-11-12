// SPDX-License-Identifier: LGPL-3.0-or-later
//! Constant-pressure one-zone nuclear burner (`burn_1_zone_const_p`)
//! coupling species mass fractions $X_i$ and $\ln T$ at prescribed total
//! pressure $P = P_{\mathrm{gas}} + P_{\mathrm{rad}}$ via `eosPT_get` and
//! Hairer & Wanner's 6-stage 4th-order L-stable `rodas4` Rosenbrock integrator
//! (`net/private/net_burn_const_p.f90` and `num/private/mod_rosenbrock.f`).

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{CRAD, LN10, QCONV};
use flexmesa_eos::{
    eos_pt_get, EosConfig, HelmTable, I_CHI_RHO, I_CHI_T, I_CP, I_ETA, I_LN_S,
};
use flexmesa_math::exp10;
use flexmesa_mtx::{dgetrf, dgetrs};
use flexmesa_num::RosenbrockVariant;
use flexmesa_rates::ScreeningMode;
use flexmesa_utils::is_bad;

use crate::burn_stepper::tableau_for;
use crate::derivs::{eval_net_with_work, with_thread_local_work};
use crate::f64x2::F64x2;
use crate::types::{NetError, NetGeneralInfo, NUM_RVS};

/// Configuration inputs for [`burn_1_zone_const_p`].
#[derive(Debug, Clone)]
pub struct BurnConstPInput<'a> {
    /// Rosenbrock solver variant (default `RosenbrockVariant::Rodas4`).
    pub solver: RosenbrockVariant,
    /// Initial temperature $T_0$ ($\mathrm{K}$).
    pub starting_temp: f64,
    /// Initial mass fractions `starting_x` (`[num_isos]`).
    pub starting_x: &'a [f64],
    /// Whether to clip negative mass fractions to `[0, 1]` during evaluation.
    pub clip: bool,
    /// Ending integration time $t_{\mathrm{end}}$ ($\mathrm{s}$).
    pub t_end: f64,
    /// Prescribed constant total pressure $P$ ($\mathrm{dyn\,cm^{-2}}$).
    pub pressure: f64,
    /// Per-reaction rate multipliers (`[num_reactions]`).
    pub rate_factors: &'a [f64],
    /// Global weak-reaction multiplier.
    pub weak_rate_factor: f64,
    /// Total reaction $Q$-values (`std_reaction_Qs`).
    pub reaction_qs: &'a [f64],
    /// Neutrino-loss $Q_{\nu}$-values (`std_reaction_neuQs`).
    pub reaction_neu_qs: &'a [f64],
    /// Plasma screening prescription (`ScreeningMode`).
    pub screening_mode: ScreeningMode,
    /// Initial step size $h_0$ ($\mathrm{s}$).
    pub h0: f64,
    /// Maximum step size $h_{\max}$ ($\mathrm{s}$, or `0.0` for `t_end`).
    pub max_step_size: f64,
    /// Maximum allowed number of steps (`max_steps`).
    pub max_steps: usize,
    /// Relative error tolerance (`rtol`, e.g., `1e-7`).
    pub rtol: f64,
    /// Absolute error tolerance (`atol`, e.g., `1e-8`).
    pub atol: f64,
    /// Lower state bound (`x_min`, e.g., `-1e-10`).
    pub x_min: f64,
}

/// Output of [`burn_1_zone_const_p`].
#[derive(Debug, Clone)]
pub struct BurnConstPOutput {
    /// Final mass fractions `ending_x` (`[num_isos]`).
    pub ending_x: Vec<f64>,
    /// Final temperature $T_{\mathrm{end}}$ ($\mathrm{K}$).
    pub ending_temp: f64,
    /// Final mass density $\rho_{\mathrm{end}}$ ($\mathrm{g\,cm^{-3}}$).
    pub ending_rho: f64,
    /// Final $\ln S_{\mathrm{end}}$.
    pub ending_lns: f64,
    /// Initial mass density $\rho_0$ ($\mathrm{g\,cm^{-3}}$).
    pub initial_rho: f64,
    /// Initial $\ln S_0$.
    pub initial_lns: f64,
    /// Number of right-hand side evaluations (`nfcn`).
    pub nfcn: usize,
    /// Number of Jacobian evaluations (`njac`).
    pub njac: usize,
    /// Number of computed steps (`nstep`).
    pub nstep: usize,
    /// Number of accepted steps (`naccpt`).
    pub naccpt: usize,
    /// Number of rejected steps (`nrejct`).
    pub nrejct: usize,
}

/// Integrates a single zone at constant total pressure $P$ over $[0, t_{\mathrm{end}}]$
/// (`burn_1_zone_const_P` in `net/private/net_burn_const_p.f90`).
///
/// # Errors
/// Returns [`NetError`] if `eosPT_get`, network evaluation, or stiff ODE
/// integration fails.
pub fn burn_1_zone_const_p(
    g: &NetGeneralInfo,
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    input: &BurnConstPInput<'_>,
) -> Result<BurnConstPOutput, NetError> {
    let num_isos = g.num_isos;
    if input.starting_x.len() != num_isos {
        return Err(NetError::IntegrationError(format!(
            "starting_x length {} != num_isos {num_isos}",
            input.starting_x.len()
        )));
    }
    let nvar = num_isos + 1;
    let mut y = vec![0.0_f64; nvar];
    y[0..num_isos].copy_from_slice(input.starting_x);
    y[num_isos] = input.starting_temp.ln();

    let nuclides = NuclideCatalog::builtin_ref();
    let mut aion = vec![0.0_f64; num_isos];
    for i in 0..num_isos {
        let cid = g.chem_id[i];
        let nuc = nuclides.get_by_id(cid).ok_or_else(|| {
            NetError::InvalidReactionOrIso(format!("unknown cid {cid}"))
        })?;
        aion[i] = nuc.z_plus_n as f64;
    }
    let mut x_buf = vec![0.0_f64; num_isos];
    let mut dydt_buf = vec![[F64x2::ZERO; NUM_RVS]; num_isos];

    let mut ending_rho = 0.0_f64;
    let mut ending_lns = 0.0_f64;
    let mut initial_rho = -1.0e99_f64;
    let mut initial_lns = -1.0e99_f64;

    let mut burn_jacob = |v: &mut [f64],
                          f: &mut [f64],
                          mut dfdv_opt: Option<&mut [f64]>|
     -> Result<(), NetError> {
        f.fill(0.0);
        if let Some(dfdv) = dfdv_opt.as_deref_mut() {
            dfdv.fill(0.0);
        }
        let mut sum_v = 0.0_f64;
        for val in v.iter_mut().take(num_isos) {
            *val = val.clamp(1.0e-30, 1.0);
            sum_v += *val;
        }
        for val in v.iter_mut().take(num_isos) {
            *val /= sum_v;
        }
        x_buf.copy_from_slice(&v[0..num_isos]);
        if input.clip {
            for val in &mut x_buf {
                *val = val.clamp(0.0, 1.0);
            }
        }

        let comp = get_composition_info(nuclides, &g.chem_id, &x_buf, false)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?;

        let log_t = v[num_isos] / LN10;
        let temp = exp10(log_t);
        let prad = CRAD * temp.powi(4) / 3.0;
        let pgas = input.pressure - prad;
        if pgas <= 0.0 || is_bad(pgas) {
            return Err(NetError::IntegrationError(format!(
                "Pgas <= 0 in burn_1_zone_const_p: Pgas={pgas}"
            )));
        }
        let lg_pgas = pgas.log10();

        let pt = eos_pt_get(ht, &g.chem_id, &x_buf, lg_pgas, log_t, eos_cfg)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?;
        let rho = pt.rho;
        let log_rho = pt.log10_rho;
        let cp = pt.eos.res[I_CP];
        let eta = pt.eos.res[I_ETA];
        let d_eta_dlnrho = pt.eos.d_dlnd[I_ETA];
        let d_eta_dlnt = pt.eos.d_dln_t[I_ETA];
        ending_rho = rho;
        ending_lns = pt.eos.res[I_LN_S];
        if initial_rho < -1.0e90 {
            initial_rho = rho;
        }
        if initial_lns < -1.0e90 {
            initial_lns = pt.eos.res[I_LN_S];
        }
        if cp <= 0.0 || is_bad(cp) {
            return Err(NetError::IntegrationError(format!(
                "Cp <= 0 in burn_1_zone_const_p: Cp={cp}"
            )));
        }

        let just_dxdt = dfdv_opt.is_none();
        with_thread_local_work(g, |w| {
            let (eps_nuc, d_eps_nuc_dt, d_eps_nuc_drho, _eps_neu) =
                eval_net_with_work(
                    g,
                    w,
                    &mut dydt_buf,
                    just_dxdt,
                    &x_buf,
                    temp,
                    log_t,
                    rho,
                    log_rho,
                    comp.abar,
                    comp.zbar,
                    comp.z2bar,
                    comp.ye,
                    eta,
                    d_eta_dlnt,
                    d_eta_dlnrho,
                    input.rate_factors,
                    input.weak_rate_factor,
                    input.screening_mode,
                    false,
                    false,
                    false,
                )?;

            for i in 0..num_isos {
                f[i] = aion[i] * dydt_buf[i][0].to_f64();
            }
            let dlnt_dt = eps_nuc / (cp * temp);
            f[num_isos] = dlnt_dt;

            if let Some(dfdv) = dfdv_opt {
                let dlnrho_dlnt_const_p =
                    -pt.eos.res[I_CHI_T] / pt.eos.res[I_CHI_RHO];
                let d_epsnuc_dlnt_const_p =
                    (d_eps_nuc_drho * rho * dlnrho_dlnt_const_p)
                        .mul_add(1.0, d_eps_nuc_dt * temp);
                let d_cp_dlnt = pt.eos.d_dlnd[I_CP]
                    .mul_add(dlnrho_dlnt_const_p, pt.eos.d_dln_t[I_CP]);

                for i in 0..num_isos {
                    let ai = aion[i];
                    for j in 0..num_isos {
                        dfdv[i * nvar + j] =
                            w.d_dydt_dy[i * num_isos + j] * ai / aion[j];
                    }
                    let d_dxdt_drho_i = ai * dydt_buf[i][2].to_f64();
                    let d_dxdt_dt_i = ai * dydt_buf[i][1].to_f64();
                    dfdv[i * nvar + num_isos] =
                        (d_dxdt_drho_i * rho * dlnrho_dlnt_const_p)
                            .mul_add(1.0, d_dxdt_dt_i * temp);
                }
                for j in 0..num_isos {
                    let d_eps_dx_j = QCONV * w.d_eps_nuc_dy[j] / aion[j];
                    dfdv[num_isos * nvar + j] = d_eps_dx_j / (cp * temp);
                }
                dfdv[num_isos * nvar + num_isos] = d_epsnuc_dlnt_const_p
                    / (cp * temp)
                    - dlnt_dt * (1.0 + d_cp_dlnt / cp);
            }
            Ok::<_, NetError>(())
        })
    };

    let tab = tableau_for(input.solver);
    let ns = tab.ns;
    let nmax = if input.max_steps == 0 {
        100_000
    } else {
        input.max_steps
    };
    let uround = 1.0e-16_f64;
    let xend = input.t_end;
    let mut x = 0.0_f64;
    let posneg = (xend - x).signum();
    let hmax = if input.max_step_size == 0.0 {
        (xend - x).abs()
    } else {
        input.max_step_size.abs()
    };
    let hmaxn = hmax.min((xend - x).abs());
    let mut h = if input.h0.abs() <= 10.0 * uround {
        1.0e-6
    } else {
        input.h0.abs()
    };
    h = h.min(hmaxn).copysign(posneg);

    let fac1 = 5.0_f64;
    let fac2 = 1.0_f64 / 3.0_f64;
    let safe = 0.9_f64;
    let y_min = input.x_min;
    let y_max = 1.0e99_f64;
    let eloi = 1.0 / tab.elo;
    let nerror = if tab.no_aux_in_error { num_isos } else { nvar };

    let mut nfcn = 0_usize;
    let mut njac = 0_usize;
    let mut nstep = 0_usize;
    let mut naccpt = 0_usize;
    let mut nrejct = 0_usize;
    let mut nsing = 0_usize;

    let mut reject = false;
    let mut last = false;
    let mut hacc = h;
    let mut erracc = 1.0e-2_f64;

    let mut dy1 = vec![0.0_f64; nvar];
    let mut dy = vec![0.0_f64; nvar];
    let mut ynew = vec![0.0_f64; nvar];
    let mut fjac = vec![0.0_f64; nvar * nvar];
    let mut e_mat = vec![0.0_f64; nvar * nvar];
    let mut ipiv = vec![0_usize; nvar];
    let mut ak = vec![vec![0.0_f64; nvar]; ns];

    'step_loop: loop {
        if nstep > nmax {
            return Err(NetError::IntegrationError(format!(
                "burn_1_zone_const_p exceeded max_steps={nmax}"
            )));
        }
        let hprev = h;
        if 0.1 * h.abs() <= x.abs() * uround {
            return Err(NetError::IntegrationError(format!(
                "step size too small in burn_1_zone_const_p: h={h}"
            )));
        }
        if last {
            break 'step_loop;
        }
        if (x + h * 1.0001 - xend) * posneg >= 0.0 {
            h = xend - x;
            last = true;
        }

        njac += 1;
        burn_jacob(&mut y, &mut dy1, Some(&mut fjac))?;

        'retry_loop: loop {
            if h <= hprev * 1.0e-30 {
                return Err(NetError::IntegrationError(
                    "too many step size reductions in burn_1_zone_const_p"
                        .to_string(),
                ));
            }
            if h <= 0.0 || is_bad(h) {
                return Err(NetError::IntegrationError(format!(
                    "invalid step size h={h}"
                )));
            }
            let fac_diag = 1.0 / (h * tab.gamma);
            for i in 0..nvar {
                for j in 0..nvar {
                    e_mat[i * nvar + j] = -fjac[i * nvar + j];
                }
                e_mat[i * nvar + i] += fac_diag;
            }
            if dgetrf(nvar, &mut e_mat, &mut ipiv).is_err() {
                nsing += 1;
                if nsing >= 5 {
                    return Err(NetError::IntegrationError(
                        "matrix repeatedly singular in burn_1_zone_const_p"
                            .to_string(),
                    ));
                }
                h *= 0.5;
                reject = true;
                last = false;
                continue 'retry_loop;
            }

            let mut stage_abort: Option<f64> = None;
            for is in 0..ns {
                if is == 0 {
                    dy.copy_from_slice(&dy1);
                    ak[is].copy_from_slice(&dy);
                } else {
                    if tab.newf[is] {
                        let mut out_of_bounds = false;
                        for j in 0..nvar {
                            let mut s = y[j];
                            for k in 0..is {
                                s = tab.ra[is][k].mul_add(ak[k][j], s);
                            }
                            ynew[j] = s;
                            if ynew[j] < y_min || ynew[j] > y_max {
                                out_of_bounds = true;
                                break;
                            }
                        }
                        if out_of_bounds {
                            stage_abort = Some(0.1);
                            break;
                        }
                        if burn_jacob(&mut ynew, &mut dy, None).is_err() {
                            stage_abort = Some(0.5);
                            break;
                        }
                    }
                    for j in 0..nvar {
                        let mut c_sum = 0.0_f64;
                        for k in 0..is {
                            c_sum =
                                (tab.rc[is][k] / h).mul_add(ak[k][j], c_sum);
                        }
                        ak[is][j] = dy[j] + c_sum;
                    }
                }
                if dgetrs(nvar, &e_mat, &ipiv, &mut ak[is]).is_err() {
                    stage_abort = Some(0.5);
                    break;
                }
            }

            if let Some(shrink) = stage_abort {
                h *= shrink;
                reject = true;
                last = false;
                continue 'retry_loop;
            }

            let mut sol_out_of_bounds = false;
            for j in 0..nvar {
                let mut s = y[j];
                for k in 0..ns {
                    s = tab.ros_m[k].mul_add(ak[k][j], s);
                }
                ynew[j] = s;
                if ynew[j] < y_min || ynew[j] > y_max {
                    sol_out_of_bounds = true;
                    break;
                }
            }
            if sol_out_of_bounds {
                h *= 0.1;
                reject = true;
                last = false;
                continue 'retry_loop;
            }

            nfcn += ns - 1;
            nstep += 1;

            for i in 0..nvar {
                let mut e_i = 0.0_f64;
                for k in 0..ns {
                    e_i = tab.ros_e[k].mul_add(ak[k][i], e_i);
                }
                dy[i] = e_i;
            }

            let mut err_sq = 0.0_f64;
            for i in 0..nerror {
                let sk =
                    input.atol + input.rtol * y[i].abs().max(ynew[i].abs());
                let r = dy[i] / sk;
                err_sq += r * r;
            }
            let mut err = (err_sq / (nerror as f64)).sqrt();
            if is_bad(err) {
                h *= 0.5;
                reject = true;
                last = false;
                continue 'retry_loop;
            }

            let mut fac = (err.powf(eloi) / safe).clamp(fac2, fac1);
            if ynew.iter().any(|&v| v < y_min || v > y_max) {
                fac = 100.0;
                err = 100.0;
            }
            let mut hnew = h / fac;
            if is_bad(hnew) {
                h *= 0.5;
                reject = true;
                last = false;
                continue 'retry_loop;
            }

            if err <= 1.0 {
                naccpt += 1;
                if naccpt > 1 {
                    let facgus = ((hacc / h) * (err * err / erracc).powf(eloi)
                        / safe)
                        .clamp(fac2, fac1);
                    fac = fac.max(facgus);
                    hnew = h / fac;
                }
                hacc = h;
                erracc = err.max(1.0e-2);
                y.copy_from_slice(&ynew);
                x += h;
                if hnew.abs() > hmaxn {
                    hnew = posneg * hmaxn;
                }
                if reject {
                    hnew = posneg * hnew.abs().min(h.abs());
                }
                reject = false;
                h = hnew;
                continue 'step_loop;
            }

            reject = true;
            last = false;
            h = hnew;
            if naccpt >= 1 {
                nrejct += 1;
            }
        }
    }

    Ok(BurnConstPOutput {
        ending_x: y[0..num_isos].to_vec(),
        ending_temp: y[num_isos].exp(),
        ending_rho,
        ending_lns,
        initial_rho,
        initial_lns,
        nfcn,
        njac,
        nstep,
        naccpt,
        nrejct,
    })
}
