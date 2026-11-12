// SPDX-License-Identifier: LGPL-3.0-or-later
//! Constant- or prescribed-$(T, \rho)$ one-zone nuclear burner (`burn_1_zone`)
//! and Bader-Deuflhard semi-implicit midpoint extrapolation integrator
//! (`netint`, `stifbs`, `simpr`, `net_pzextr`) from
//! `net/private/net_burn.f90` and `net/private/net_burn_support.f90`.

use flexmesa_chem::{get_composition_info, NuclideCatalog, NUM_CATEGORIES};
use flexmesa_const::QCONV;
use flexmesa_eos::{eos_dt_get, EosConfig, HelmTable, I_ETA};
use flexmesa_interp1d::{interp_pm, interp_value};
use flexmesa_math::exp10;
use flexmesa_rates::ScreeningMode;
use flexmesa_utils::is_bad;

use crate::burn_stepper::netint;
use crate::derivs::{eval_net_with_work, with_thread_local_work};
use crate::f64x2::F64x2;
use crate::types::{NetError, NetGeneralInfo, NUM_RVS};

/// Configuration and history inputs for [`burn_1_zone`].
#[derive(Debug, Clone)]
pub struct Burn1ZoneInput<'a> {
    /// Starting integration time $t_{\mathrm{start}}$ ($\mathrm{s}$).
    pub t_start: f64,
    /// Ending integration time $t_{\mathrm{end}}$ ($\mathrm{s}$).
    pub t_end: f64,
    /// Initial mass fractions `starting_x` (`[species]`).
    pub starting_x: &'a [f64],
    /// Time knot grid `times` (`[ntimes]`).
    pub times: &'a [f64],
    /// $\log_{10}(T / \mathrm{K})$ at each time knot (`[ntimes]`).
    pub log10_ts: &'a [f64],
    /// $\log_{10}(\rho / \mathrm{g\,cm^{-3}})$ at each time knot (`[ntimes]`).
    pub log10_rhos: &'a [f64],
    /// Electron degeneracy $\eta$ at each time knot (`[ntimes]`).
    pub etas: &'a [f64],
    /// Optional external mass-fraction source term `dxdt_source_term` (`[species]`).
    pub dxdt_source_term: Option<&'a [f64]>,
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
    /// Relative integration error tolerance (`eps`, e.g., `1e-8`).
    pub eps: f64,
    /// Abundance scaling floor (`odescal`, e.g., `1e-12`).
    pub odescal: f64,
}

/// Output of [`burn_1_zone`].
#[derive(Debug, Clone)]
pub struct Burn1ZoneOutput {
    /// Final mass fractions `ending_x` (`[species]`).
    pub ending_x: Vec<f64>,
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

pub(crate) struct OneZoneStepBuf {
    pub(crate) last_eval: Option<([f64; NUM_CATEGORIES], f64)>,
    pub(crate) x_buf: Vec<f64>,
    pub(crate) dydt_buf: Vec<[F64x2; NUM_RVS]>,
}

impl OneZoneStepBuf {
    pub(crate) fn new(species: usize) -> Self {
        Self {
            last_eval: None,
            x_buf: vec![0.0_f64; species],
            dydt_buf: vec![[F64x2::ZERO; NUM_RVS]; species],
        }
    }
}

pub(crate) fn collect_aion(
    nuclides: &NuclideCatalog,
    chem_id: &[usize],
) -> Result<Vec<f64>, NetError> {
    chem_id
        .iter()
        .map(|&cid| {
            nuclides
                .get_by_id(cid)
                .map(|nuc| nuc.z_plus_n as f64)
                .ok_or_else(|| {
                    NetError::InvalidReactionOrIso(format!("unknown cid {cid}"))
                })
        })
        .collect()
}

pub(crate) fn unpack_clamped_molar_y(
    time: f64,
    y: &mut [f64],
    aion: &[f64],
    x_buf: &mut [f64],
) -> Result<(), NetError> {
    for (i, &ai) in aion.iter().enumerate() {
        if is_bad(y[i]) {
            return Err(NetError::IntegrationError(format!(
                "bad y[{i}]={} at time={time}",
                y[i]
            )));
        }
        y[i] = y[i].clamp(1.0e-30, 1.0);
        x_buf[i] = y[i] * ai;
    }
    Ok(())
}

pub(crate) fn compute_post_burn_energetics(
    nuclides: &NuclideCatalog,
    chem_id: &[usize],
    aion: &[f64],
    starting_x: &[f64],
    ending_y: &[f64],
    dt: f64,
    eps_neu_total: f64,
) -> (Vec<f64>, f64) {
    let species = aion.len();
    let mut ending_x = vec![0.0_f64; species];
    let mut burn_ergs = 0.0_f64;
    for i in 0..species {
        ending_x[i] = ending_y[i] * aion[i];
        let dx = ending_x[i] - starting_x[i];
        let q_iso = nuclides.get_q(chem_id[i]).unwrap_or(0.0);
        burn_ergs += q_iso * dx / aion[i];
    }
    burn_ergs *= QCONV;
    let avg_eps_nuc = burn_ergs / dt - eps_neu_total;
    (ending_x, avg_eps_nuc)
}

/// Integrates a single zone over $[t_{\mathrm{start}}, t_{\mathrm{end}}]$ along
/// a constant or piecewise-monotonic cubic $(T(t), \rho(t))$ trajectory using
/// the Bader-Deuflhard semi-implicit midpoint rule (`burn_1_zone` in
/// `net/private/net_burn.f90`).
///
/// # Errors
/// Returns [`NetError`] if spline construction, EOS evaluation, or stiff ODE
/// integration fails.
pub fn burn_1_zone(
    g: &NetGeneralInfo,
    ht: &HelmTable,
    eos_cfg: &EosConfig,
    input: &Burn1ZoneInput<'_>,
) -> Result<Burn1ZoneOutput, NetError> {
    let species = g.num_isos;
    if input.starting_x.len() != species {
        return Err(NetError::IntegrationError(format!(
            "starting_x length {} != species {species}",
            input.starting_x.len()
        )));
    }
    let ntimes = input.times.len();
    if ntimes == 0
        || input.log10_ts.len() != ntimes
        || input.log10_rhos.len() != ntimes
    {
        return Err(NetError::IntegrationError(
            "invalid times / log10_ts / log10_rhos lengths".to_string(),
        ));
    }

    let log10_ts_coeffs = if ntimes > 1 {
        interp_pm(input.times, input.log10_ts)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?
    } else {
        vec![[input.log10_ts[0], 0.0, 0.0, 0.0]]
    };
    let log10_rhos_coeffs = if ntimes > 1 {
        interp_pm(input.times, input.log10_rhos)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?
    } else {
        vec![[input.log10_rhos[0], 0.0, 0.0, 0.0]]
    };

    let nuclides = NuclideCatalog::builtin_ref();
    let aion = collect_aion(nuclides, &g.chem_id)?;
    let mut ending_y: Vec<f64> = input
        .starting_x
        .iter()
        .zip(&aion)
        .map(|(&xi, &ai)| xi / ai)
        .collect();

    let start = input.t_start;
    let stopp = input.t_end;
    let stptry = if input.stptry_in > 0.0 {
        input.stptry_in
    } else {
        (start * 1.0e-10).max(1.0e-16)
    };
    let stpmin = stptry * 1.0e-12;

    let mut step_buf = OneZoneStepBuf::new(species);
    let mut dens_dfdy = vec![0.0_f64; species * species];
    let mut dmat = vec![0.0_f64; species * species];

    let mut nfcn = 0_usize;
    let mut njac = 0_usize;

    let eval_step = |time: f64,
                     y: &mut [f64],
                     f_out: Option<&mut [f64]>,
                     dfdy_out: Option<&mut [f64]>,
                     buf: &mut OneZoneStepBuf|
     -> Result<(), NetError> {
        let (lg_t, lg_rho) = if ntimes == 1 {
            (input.log10_ts[0], input.log10_rhos[0])
        } else {
            let lgt = interp_value(input.times, &log10_ts_coeffs, time)
                .map_err(|e| NetError::IntegrationError(e.to_string()))?;
            let lgr = interp_value(input.times, &log10_rhos_coeffs, time)
                .map_err(|e| NetError::IntegrationError(e.to_string()))?;
            (lgt, lgr)
        };

        unpack_clamped_molar_y(time, y, &aion, &mut buf.x_buf)?;

        let rho = exp10(lg_rho);
        let temp = exp10(lg_t);

        let comp =
            get_composition_info(nuclides, &g.chem_id, &buf.x_buf, false)
                .map_err(|e| NetError::IntegrationError(e.to_string()))?;

        let eos = eos_dt_get(ht, rho, temp, &g.chem_id, &buf.x_buf, eos_cfg)
            .map_err(|e| NetError::IntegrationError(e.to_string()))?;
        let eta = eos.res[I_ETA];
        let d_eta_dlnt = eos.d_dln_t[I_ETA];
        let d_eta_dlnrho = eos.d_dlnd[I_ETA];

        let dxdt_only = dfdy_out.is_none();
        let (cats, eps_neu_total) = with_thread_local_work(g, |w| {
            let (_eps_nuc, _deps_dt, _deps_drho, eps_neu) = eval_net_with_work(
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
                    let src = input.dxdt_source_term.map_or(0.0, |s| s[j]);
                    f[j] = buf.dydt_buf[j][0].to_f64() + src / aion[j];
                }
            }
            if let Some(dfdy) = dfdy_out {
                dfdy[..species * species]
                    .copy_from_slice(&w.d_dydt_dy[..species * species]);
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
        species,
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
                "no eval_net result in burn_1_zone".to_string(),
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

    Ok(Burn1ZoneOutput {
        ending_x,
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
