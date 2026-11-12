// SPDX-License-Identifier: LGPL-3.0-or-later
//! Constant-density variable-temperature one-zone nuclear burner
//! (`burn_1_zone_const_density`) coupling molar abundances $Y_i = X_i / A_i$
//! and $\ln T$ at fixed mass density $\rho$ via `eosDT_get` and the
//! Bader-Deuflhard semi-implicit midpoint integrator (`netint`) from
//! `net/private/net_burn_const_density.f90`.

use flexmesa_chem::{get_composition_info, NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::{LN10, QCONV};
use flexmesa_eos::{eos_dt_get, EosConfig, HelmTable, I_CV, I_ETA};
use flexmesa_math::exp10;
use flexmesa_rates::ScreeningMode;
use flexmesa_utils::is_bad;

use crate::burn::{
    collect_aion, compute_post_burn_energetics, unpack_clamped_molar_y,
    OneZoneStepBuf,
};
use crate::burn_stepper::netint;
use crate::derivs::{eval_net_with_work, with_thread_local_work};
use crate::types::{NetError, NetGeneralInfo};

/// Configuration inputs for [`burn_1_zone_const_density`].
#[derive(Debug, Clone)]
pub struct BurnConstDensityInput<'a> {
    /// Starting integration time $t_{\mathrm{start}}$ ($\mathrm{s}$).
    pub t_start: f64,
    /// Ending integration time $t_{\mathrm{end}}$ ($\mathrm{s}$).
    pub t_end: f64,
    /// Initial mass fractions `starting_x` (`[species]`).
    pub starting_x: &'a [f64],
    /// Initial $\log_{10}(T_0 / \mathrm{K})$ (`starting_log10T`).
    pub starting_log10_t: f64,
    /// Fixed $\log_{10}(\rho / \mathrm{g\,cm^{-3}})$ (`log10Rho`).
    pub log10_rho: f64,
    /// Per-reaction rate multipliers (`[num_reactions]`).
    pub rate_factors: &'a [f64],
    /// Global weak-reaction multiplier.
    pub weak_rate_factor: f64,
    /// Plasma screening prescription (`ScreeningMode`).
    pub screening_mode: ScreeningMode,
    /// Suggested initial step size (`stptry_in`, or `0.0` for automatic).
    pub stptry_in: f64,
    /// Maximum allowed number of integration substeps (`max_steps`).
    pub max_steps: usize,
    /// Relative integration error tolerance (`eps`, e.g., `1e-5`).
    pub eps: f64,
    /// Abundance scaling floor (`odescal`, e.g., `1e-5`).
    pub odescal: f64,
}

/// Output of [`burn_1_zone_const_density`].
#[derive(Debug, Clone)]
pub struct BurnConstDensityOutput {
    /// Final mass fractions `ending_x` (`[species]`).
    pub ending_x: Vec<f64>,
    /// Final $\log_{10}(T_{\mathrm{end}} / \mathrm{K})$ (`ending_log10T`).
    pub ending_log10_t: f64,
    /// Final category-resolved nuclear energy generation rates (`[NUM_CATEGORIES]`).
    pub eps_nuc_categories: [f64; NUM_CATEGORIES],
    /// Time-averaged nuclear energy generation rate ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub avg_eps_nuc: f64,
    /// Final thermal + weak neutrino loss rate ($\mathrm{erg\,g^{-1}\,s^{-1}}$).
    pub eps_neu_total: f64,
    /// Total number of right-hand side evaluations (`nfcn`).
    pub nfcn: usize,
    /// Total number of Jacobian evaluations (`njac`).
    pub njac: usize,
    /// Total number of substeps taken (`nstep`).
    pub nstep: usize,
    /// Number of accepted steps without retry (`naccpt`).
    pub naccpt: usize,
    /// Number of retried steps (`nrejct`).
    pub nrejct: usize,
}

/// Integrates a single zone over $[t_{\mathrm{start}}, t_{\mathrm{end}}]$ at
/// constant mass density $\rho = 10^{\log_{10}\rho}$ while co-evolving
/// $\mathrm{d}\ln T/\mathrm{d}t = \epsilon_{\mathrm{nuc}} / (C_V T)$ using the
/// Bader-Deuflhard semi-implicit midpoint rule (`burn_const_density_1_zone` in
/// `net/private/net_burn_const_density.f90:74-500`).
///
/// # Errors
/// Returns [`NetError`] if EOS evaluation, network evaluation, or stiff ODE
/// integration fails.
pub fn burn_1_zone_const_density(
    g: &NetGeneralInfo,
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    input: &BurnConstDensityInput<'_>,
) -> Result<BurnConstDensityOutput, NetError> {
    let species = g.num_isos;
    if input.starting_x.len() != species {
        return Err(NetError::IntegrationError(format!(
            "starting_x length {} != species {species}",
            input.starting_x.len()
        )));
    }
    let nvar = species + 1;
    let nuclides = NuclideCatalog::builtin_ref();
    let aion = collect_aion(nuclides, &g.chem_id)?;
    let mut ending_y = vec![0.0_f64; nvar];
    for i in 0..species {
        ending_y[i] = input.starting_x[i] / aion[i];
    }
    ending_y[species] = input.starting_log10_t * LN10;

    let lg_rho = input.log10_rho;
    let rho = exp10(lg_rho);
    let start = input.t_start;
    let stopp = input.t_end;
    let stptry = if input.stptry_in > 0.0 {
        input.stptry_in
    } else {
        (start * 1.0e-10).max(1.0e-16)
    };
    let stpmin = stptry * 1.0e-12;

    let mut step_buf = OneZoneStepBuf::new(species);
    let mut dens_dfdy = vec![0.0_f64; nvar * nvar];
    let mut dmat = vec![0.0_f64; nvar * nvar];

    let mut nfcn = 0_usize;
    let mut njac = 0_usize;

    let eval_step = |time: f64,
                     y: &mut [f64],
                     f_out: Option<&mut [f64]>,
                     dfdy_out: Option<&mut [f64]>,
                     buf: &mut OneZoneStepBuf|
     -> Result<(), NetError> {
        unpack_clamped_molar_y(time, y, &aion, &mut buf.x_buf)?;
        if is_bad(y[species]) {
            return Err(NetError::IntegrationError(format!(
                "bad lnT={} at time={time}",
                y[species]
            )));
        }

        let lg_t = y[species] / LN10;
        let temp = exp10(lg_t);

        let comp =
            get_composition_info(nuclides, &g.chem_id, &buf.x_buf, false)
                .map_err(|e| NetError::IntegrationError(e.to_string()))?;

        let eos = eos_dt_get(ht, rho, temp, &g.chem_id, &buf.x_buf, eos_cfg)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?;
        let cv = eos.res[I_CV];
        if cv <= 0.0 || is_bad(cv) {
            return Err(NetError::IntegrationError(format!(
                "Cv <= 0 in burn_1_zone_const_density: Cv={cv}"
            )));
        }
        let d_cv_dlnt = eos.d_dln_t[I_CV];
        let eta = eos.res[I_ETA];
        let d_eta_dlnt = eos.d_dln_t[I_ETA];
        let d_eta_dlnrho = 0.0;

        let dxdt_only = dfdy_out.is_none();
        let (cats, eps_neu_total) = with_thread_local_work(g, |w| {
            let (eps_nuc, d_eps_nuc_dt, _deps_drho, eps_neu) =
                eval_net_with_work(
                    g,
                    w,
                    &mut buf.dydt_buf,
                    dxdt_only,
                    &buf.x_buf,
                    temp,
                    lg_t,
                    rho,
                    lg_rho,
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
            if let Some(f) = f_out {
                for j in 0..species {
                    f[j] = buf.dydt_buf[j][0].to_f64();
                }
                f[species] = eps_nuc / (cv * temp);
            }
            if let Some(dfdy) = dfdy_out {
                for i in 0..species {
                    let row_n = i * nvar;
                    let row_s = i * species;
                    dfdy[row_n..row_n + species]
                        .copy_from_slice(&w.d_dydt_dy[row_s..row_s + species]);
                    // Exact upstream `net_burn_const_density.f90:476` (`dfdy(j,nvar) = 0d0`).
                    dfdy[row_n + species] = 0.0;
                }
                for j in 0..species {
                    let d_eps_dy_j = QCONV * w.d_eps_nuc_dy[j];
                    dfdy[species * nvar + j] = d_eps_dy_j / (cv * temp);
                }
                let df_t_t = d_eps_nuc_dt / cv
                    - (1.0 + d_cv_dlnt / cv) * eps_nuc / (cv * temp);
                if is_bad(df_t_t) {
                    return Err(NetError::IntegrationError(format!(
                        "bad dfdy[nvar,nvar]={df_t_t} in burn_1_zone_const_density"
                    )));
                }
                dfdy[species * nvar + species] = df_t_t;
            }
            let mut cats = [0.0_f64; NUM_CATEGORIES];
            for c in 0..NUM_CATEGORIES {
                cats[c] = QCONV * w.eps_nuc_categories[c];
            }
            Ok::<_, NetError>((cats, eps_neu))
        })?;

        buf.last_eval = Some((cats, eps_neu_total));
        Ok(())
    };

    let (naccpt, nrejct, nstep) = netint(
        start,
        stptry,
        stpmin,
        input.max_steps,
        stopp,
        &mut ending_y,
        input.eps,
        species,
        nvar,
        input.odescal,
        &mut dens_dfdy,
        &mut dmat,
        &mut step_buf,
        &mut |time, y, f, w| {
            nfcn += 1;
            eval_step(time, y, Some(f), None, w)
        },
        &mut |time, y, dfdy, w| {
            njac += 1;
            eval_step(time, y, None, Some(dfdy), w)
        },
    )?;

    let (eps_nuc_categories, eps_neu_total) =
        step_buf.last_eval.ok_or_else(|| {
            NetError::IntegrationError(
                "no eval_net result in burn_1_zone_const_density".to_string(),
            )
        })?;
    let (ending_x, avg_eps_nuc) = compute_post_burn_energetics(
        nuclides,
        &g.chem_id,
        &aion,
        input.starting_x,
        &ending_y,
        stopp - start,
        eps_neu_total,
    );
    let ending_log10_t = ending_y[species] / LN10;

    Ok(BurnConstDensityOutput {
        ending_x,
        ending_log10_t,
        eps_nuc_categories,
        avg_eps_nuc,
        eps_neu_total,
        nfcn,
        njac,
        nstep,
        naccpt,
        nrejct,
    })
}

#[cfg(test)]
mod tests {
    use flexmesa_eos::I_LN_E;

    use super::*;
    use crate::{F64x2, NUM_RVS};

    #[test]
    fn test_burn_1_zone_const_density_first_law_energy_and_jacobian_parity() {
        let g = NetGeneralInfo::from_name("basic.net").expect("basic.net");
        let ht = HelmTable::builtin();
        let eos_cfg = EosConfig::default();
        let nuclides = NuclideCatalog::builtin_ref();
        let species = g.num_isos;

        let mut x0 = vec![0.0_f64; species];
        let idx_of = |name: &str| {
            g.chem_id
                .iter()
                .position(|&c| {
                    nuclides.get_by_id(c).is_some_and(|n| n.name == name)
                })
                .expect("isotope in basic.net")
        };
        let i_he4 = idx_of("he4");
        let i_c12 = idx_of("c12");
        let i_o16 = idx_of("o16");
        x0[i_he4] = 0.85;
        x0[i_c12] = 0.10;
        x0[i_o16] = 0.05;

        let rate_factors = vec![1.0_f64; g.num_reactions];
        let dt = 10.0;
        let lg_rho = 4.0;
        let rho = exp10(lg_rho);
        let lg_t0 = 8.50;
        let input = BurnConstDensityInput {
            t_start: 0.0,
            t_end: dt,
            starting_x: &x0,
            starting_log10_t: lg_t0,
            log10_rho: lg_rho,
            rate_factors: &rate_factors,
            weak_rate_factor: 0.0,
            screening_mode: ScreeningMode::Extended,
            stptry_in: 1.0e-6,
            max_steps: 1000,
            eps: 1.0e-9,
            odescal: 1.0e-10,
        };
        let out = burn_1_zone_const_density(&g, ht, &eos_cfg, &input)
            .expect("burn_1_zone_const_density succeeds");

        let x_sum: f64 = out.ending_x.iter().sum();
        assert!(
            (x_sum - 1.0).abs() < 1.0e-12,
            "baryon sum {x_sum} deviates from 1.0"
        );
        assert!(out.ending_x[i_he4] < x0[i_he4]);
        assert!(out.ending_log10_t > lg_t0);
        assert!(out.avg_eps_nuc > 0.0);

        // Tier 1 First-Law Oracle: int_{T0}^{T1} Cv(rho, T, X_mid) dT == dt * avg_eps_nuc
        let x_mid: Vec<f64> = (0..species)
            .map(|i| 0.5 * (x0[i] + out.ending_x[i]))
            .collect();
        let e0 =
            eos_dt_get(ht, rho, exp10(lg_t0), &g.chem_id, &x_mid, &eos_cfg)
                .unwrap()
                .res[I_LN_E]
                .exp();
        let e1 = eos_dt_get(
            ht,
            rho,
            exp10(out.ending_log10_t),
            &g.chem_id,
            &x_mid,
            &eos_cfg,
        )
        .unwrap()
        .res[I_LN_E]
            .exp();
        let de_eos = e1 - e0;
        let de_nuc = dt * out.avg_eps_nuc;
        let rel_err = (de_eos - de_nuc).abs() / de_nuc.abs();
        assert!(
            rel_err <= 2.0e-4,
            "first-law Cv*dT integral mismatch: de_eos={de_eos:.8e}, de_nuc={de_nuc:.8e}, rel={rel_err:.3e}"
        );

        // Tier 1 Central-FD Jacobian verification on the (species + 1) state vector
        let aion: Vec<f64> = g
            .chem_id
            .iter()
            .map(|&c| nuclides.get_by_id(c).unwrap().z_plus_n as f64)
            .collect();
        let nvar = species + 1;
        let mut y_base: Vec<f64> =
            (0..species).map(|i| x0[i] / aion[i]).collect();
        y_base.push(lg_t0 * LN10);

        let eval_rhs = |y: &[f64],
                        freeze_comp: Option<(
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
            f64,
        )>| {
            let x_cur: Vec<f64> =
                (0..species).map(|i| y[i] * aion[i]).collect();
            let lg_t = y[species] / LN10;
            let temp = exp10(lg_t);
            let comp =
                get_composition_info(nuclides, &g.chem_id, &x_cur, false)
                    .unwrap();
            let eos = eos_dt_get(ht, rho, temp, &g.chem_id, &x_cur, &eos_cfg)
                .unwrap();
            let (abar, zbar, z2bar, ye, cv, eta, d_eta_dlnt) = freeze_comp
                .unwrap_or((
                    comp.abar,
                    comp.zbar,
                    comp.z2bar,
                    comp.ye,
                    eos.res[I_CV],
                    eos.res[I_ETA],
                    eos.d_dln_t[I_ETA],
                ));
            let d_cv_dlnt = eos.d_dln_t[I_CV];
            let mut dydt_buf = vec![[F64x2::ZERO; NUM_RVS]; species];
            let mut f = vec![0.0; nvar];
            let mut dfdy = vec![0.0; nvar * nvar];
            with_thread_local_work(&g, |w| {
                let (eps_nuc, d_eps_dt, _, _) = eval_net_with_work(
                    &g,
                    w,
                    &mut dydt_buf,
                    false,
                    &x_cur,
                    temp,
                    lg_t,
                    rho,
                    lg_rho,
                    abar,
                    zbar,
                    z2bar,
                    ye,
                    eta,
                    d_eta_dlnt,
                    0.0,
                    &rate_factors,
                    0.0,
                    ScreeningMode::Extended,
                    false,
                    false,
                    false,
                )
                .unwrap();
                for j in 0..species {
                    f[j] = dydt_buf[j][0].to_f64();
                }
                f[species] = eps_nuc / (cv * temp);
                for i in 0..species {
                    dfdy[i * nvar..i * nvar + species].copy_from_slice(
                        &w.d_dydt_dy[i * species..(i + 1) * species],
                    );
                }
                for j in 0..species {
                    dfdy[species * nvar + j] =
                        (QCONV * w.d_eps_nuc_dy[j]) / (cv * temp);
                }
                dfdy[species * nvar + species] = d_eps_dt / cv
                    - (1.0 + d_cv_dlnt / cv) * eps_nuc / (cv * temp);
            });
            (f, dfdy, (abar, zbar, z2bar, ye, cv, eta, d_eta_dlnt))
        };

        let (f_base, dfdy, frozen) = eval_rhs(&y_base, None);
        // Check (d lnT/dt) / d lnT entry via central FD wrt lnT
        let h_lnt = 1.0e-6;
        let mut y_p = y_base.clone();
        let mut y_m = y_base.clone();
        y_p[species] += h_lnt;
        y_m[species] -= h_lnt;
        let (f_p, _, _) = eval_rhs(&y_p, None);
        let (f_m, _, _) = eval_rhs(&y_m, None);
        let fd_tt = (f_p[species] - f_m[species]) / (2.0 * h_lnt);
        let an_tt = dfdy[species * nvar + species];
        assert!(
            (an_tt - fd_tt).abs() / an_tt.abs().max(fd_tt.abs()) <= 1.0e-5,
            "df_lnT/dlnT mismatch: analytic={an_tt:.8e}, fd={fd_tt:.8e}"
        );

        // Check molar abundance columns (0..species) at frozen macro moments/Cv
        for &j in &[i_he4, i_c12, i_o16] {
            let h_y = 1.0e-6 * y_base[j].max(1.0e-3);
            let mut yp = y_base.clone();
            let mut ym = y_base.clone();
            yp[j] += h_y;
            ym[j] -= h_y;
            let (fp, _, _) = eval_rhs(&yp, Some(frozen));
            let (fm, _, _) = eval_rhs(&ym, Some(frozen));
            for i in 0..=species {
                let fd_ij = (fp[i] - fm[i]) / (2.0 * h_y);
                let an_ij = dfdy[i * nvar + j];
                let scale = an_ij.abs().max(fd_ij.abs());
                let roundoff_floor =
                    32.0 * f64::EPSILON * f_base[i].abs() / h_y;
                assert!(
                    (an_ij - fd_ij).abs() <= roundoff_floor.max(1.0e-18)
                        || (an_ij - fd_ij).abs() <= 1.0e-6 * scale,
                    "df[{i}]/dy[{j}] mismatch: analytic={an_ij:.8e}, fd={fd_ij:.8e}"
                );
            }
        }
    }
}
