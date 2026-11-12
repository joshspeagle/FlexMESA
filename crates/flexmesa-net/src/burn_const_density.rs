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

use crate::burn_stepper::netint;
use crate::derivs::{eval_net_with_work, with_thread_local_work};
use crate::f64x2::F64x2;
use crate::types::{NetError, NetGeneralInfo, NUM_RVS};

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
    /// Total reaction $Q$-values (`std_reaction_Qs`).
    pub reaction_qs: &'a [f64],
    /// Neutrino-loss $Q_{\nu}$-values (`std_reaction_neuQs`).
    pub reaction_neu_qs: &'a [f64],
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
    let mut aion = vec![0.0_f64; species];
    let mut save_x = vec![0.0_f64; species];
    let mut ending_y = vec![0.0_f64; nvar];
    for i in 0..species {
        let cid = g.chem_id[i];
        let nuc = nuclides.get_by_id(cid).ok_or_else(|| {
            NetError::InvalidReactionOrIso(format!("unknown cid {cid}"))
        })?;
        aion[i] = nuc.z_plus_n as f64;
        save_x[i] = input.starting_x[i];
        ending_y[i] = input.starting_x[i] / aion[i];
    }
    ending_y[species] = input.starting_log10_t * LN10;

    let lg_rho = input.log10_rho;
    let rho = exp10(lg_rho);
    let start = input.t_start;
    let stopp = input.t_end;
    let stptry = (start * 1.0e-10).max(1.0e-16);
    let stpmin = stptry * 1.0e-12;

    struct StepBuf {
        last_eval: Option<([f64; NUM_CATEGORIES], f64)>,
        x_buf: Vec<f64>,
        dydt_buf: Vec<[F64x2; NUM_RVS]>,
    }
    let mut step_buf = StepBuf {
        last_eval: None,
        x_buf: vec![0.0_f64; species],
        dydt_buf: vec![[F64x2::ZERO; NUM_RVS]; species],
    };
    let mut dens_dfdy = vec![0.0_f64; nvar * nvar];
    let mut dmat = vec![0.0_f64; nvar * nvar];

    let mut nfcn = 0_usize;
    let mut njac = 0_usize;

    let eval_step = |time: f64,
                     y: &mut [f64],
                     f_out: Option<&mut [f64]>,
                     dfdy_out: Option<&mut [f64]>,
                     buf: &mut StepBuf|
     -> Result<(), NetError> {
        for i in 0..species {
            if is_bad(y[i]) {
                return Err(NetError::IntegrationError(format!(
                    "bad y[{i}]={} at time={time}",
                    y[i]
                )));
            }
            y[i] = y[i].clamp(1.0e-30, 1.0);
            buf.x_buf[i] = y[i] * aion[i];
        }
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
                    let d_eps_dx_j = QCONV * w.d_eps_nuc_dy[j] / aion[j];
                    dfdy[species * nvar + j] = d_eps_dx_j / (cv * temp);
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

    let mut ending_x = vec![0.0_f64; species];
    let mut burn_ergs = 0.0_f64;
    for i in 0..species {
        ending_x[i] = ending_y[i] * aion[i];
        let dx = ending_x[i] - save_x[i];
        let cid = g.chem_id[i];
        let q_iso = nuclides.get_q(cid).unwrap_or(0.0);
        burn_ergs += q_iso * dx / aion[i];
    }
    burn_ergs *= QCONV;
    let (eps_nuc_categories, eps_neu_total) =
        step_buf.last_eval.ok_or_else(|| {
            NetError::IntegrationError(
                "no eval_net result in burn_1_zone_const_density".to_string(),
            )
        })?;
    let avg_eps_nuc = burn_ergs / (stopp - start) - eps_neu_total;
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
    use super::*;

    #[test]
    fn test_burn_1_zone_const_density_he_ignition_heats_and_conserves_baryons()
    {
        let g = NetGeneralInfo::from_name("basic.net").expect("basic.net");
        let ht = HelmTable::builtin();
        let eos_cfg = EosConfig::default();
        // Pure He4 at logT = 8.3, logRho = 4.0 over dt = 1e4 s
        let mut x0 = vec![0.0_f64; g.num_isos];
        let i_he4 = g
            .chem_id
            .iter()
            .position(|&c| {
                NuclideCatalog::builtin_ref()
                    .get_by_id(c)
                    .is_some_and(|n| n.name == "he4")
            })
            .expect("he4 in basic.net");
        x0[i_he4] = 1.0;
        let rate_factors = vec![1.0_f64; g.num_reactions];
        let qs = vec![0.0_f64; g.num_reactions];
        let input = BurnConstDensityInput {
            t_start: 0.0,
            t_end: 1.0e4,
            starting_x: &x0,
            starting_log10_t: 8.3,
            log10_rho: 4.0,
            rate_factors: &rate_factors,
            weak_rate_factor: 1.0,
            reaction_qs: &qs,
            reaction_neu_qs: &qs,
            screening_mode: ScreeningMode::Extended,
            stptry_in: 0.0,
            max_steps: 500,
            eps: 1.0e-6,
            odescal: 1.0e-8,
        };
        let out = burn_1_zone_const_density(&g, ht, &eos_cfg, &input)
            .expect("burn_1_zone_const_density succeeds");
        let x_sum: f64 = out.ending_x.iter().sum();
        assert!(
            (x_sum - 1.0).abs() < 1.0e-12,
            "baryon sum {x_sum} deviates from 1.0"
        );
        assert!(out.ending_x[i_he4] < 1.0, "he4 should burn into c12/o16");
        assert!(
            out.ending_log10_t > 8.3,
            "exothermic 3-alpha burning must raise log10(T): {}",
            out.ending_log10_t
        );
        assert!(out.avg_eps_nuc > 0.0);
    }
}
