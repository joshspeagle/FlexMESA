// SPDX-License-Identifier: GPL-3.0-or-later
//! General reaction-network RHS, analytical Jacobians, and energy generation
//! (`net_derivs.f90` and `net_eval.f90`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::QCONV;
use flexmesa_rates::ScreeningMode;

use crate::approx21_eval::{eval_net_approx21_into, eval_net_approx21_procs};
use crate::derivs_paired::{
    get_basic_2_to_1_derivs, get_basic_2_to_2_derivs,
    get_general_1_to_1_derivs, get_general_2_to_1_derivs,
    get_general_2_to_2_derivs,
};
use crate::derivs_support::{
    do_in_out, do_in_out_neu, get_weaklib_rates, update_special_rates,
    NetWorkState,
};
use crate::screen::screen_net;
use crate::types::{
    NetError, NetEvalResult, NetGeneralInfo, ReaclibKind, IH1, IPROTON,
    NUM_CATEGORIES, NUM_RVS, RATTAB_TEMP_LO, RATTAB_TLO,
};
use flexmesa_math::Quad;

const MIN_Z_FOR_SWITCH_TO_PROT: i32 = 12;

fn nuclide_aion(
    nuclides: &NuclideCatalog,
    cid: usize,
) -> Result<f64, NetError> {
    nuclides
        .get_by_id(cid)
        .map(|n| n.z_plus_n as f64)
        .ok_or_else(|| NetError::InvalidReactionOrIso(format!("id={cid}")))
}

#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
fn eval_net_setup(
    g: &NetGeneralInfo, w: &mut NetWorkState, x: &[f64],
    temp: &mut f64, log_t: &mut f64, rho: f64,
    abar: f64, zbar: f64, z2bar: f64, ye: f64,
    eta: f64, d_eta_dln_t: f64, d_eta_dln_rho: f64,
    rate_factors: &[f64], weak_rate_factor: f64,
    use_suzuki: bool, do_ecapture: bool,
) -> Result<(), NetError> {
    if x.len() != g.num_isos {
        return Err(NetError::IntegrationError(format!(
            "expected {} mass fractions, got {}",
            g.num_isos, x.len()
        )));
    }
    if *log_t < RATTAB_TLO {
        *log_t = RATTAB_TLO;
        *temp = RATTAB_TEMP_LO;
    }
    let nuclides = NuclideCatalog::builtin_ref();
    for (i, &cid) in g.chem_id.iter().enumerate() {
        let a = nuclide_aion(nuclides, cid)?;
        w.y[i] = x[i].clamp(0.0, 1.0) / a;
    }
    if g.num_wk_reactions > 0 {
        get_weaklib_rates(
            g, w, *temp, rho, ye, abar, zbar, z2bar, eta, d_eta_dln_t,
            d_eta_dln_rho, weak_rate_factor, use_suzuki, do_ecapture,
        );
    }
    g.eval_raw_rates_from_tables(
        ye, *log_t, *temp, rho, rate_factors,
        &mut w.rate_raw[..g.num_reactions],
        &mut w.rate_raw_dt[..g.num_reactions],
        &mut w.rate_raw_drho[..g.num_reactions],
    )
}

/// Evaluates a nuclear reaction network at `(x, temp, log_t, rho, log_rho)`
/// (`eval_net` in `net/private/net_eval.f90`).
#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
pub fn eval_net(
    g: &NetGeneralInfo, rates_only: bool, just_dxdt: bool, x: &[f64],
    temp: f64, log_t: f64, rho: f64, log_rho: f64,
    abar: f64, zbar: f64, z2bar: f64, ye: f64,
    eta: f64, d_eta_dln_t: f64, d_eta_dln_rho: f64,
    rate_factors: &[f64], weak_rate_factor: f64,
    screening_mode: ScreeningMode, use_3a_fl87: bool,
    use_suzuki: bool, do_ecapture: bool, symbolic: bool,
) -> Result<NetEvalResult, NetError> {
    let mut w = NetWorkState::new(g);
    eval_net_result_with_work(
        g, &mut w, rates_only, just_dxdt, x, temp, log_t, rho, log_rho, abar,
        zbar, z2bar, ye, eta, d_eta_dln_t, d_eta_dln_rho, rate_factors,
        weak_rate_factor, screening_mode, use_3a_fl87, use_suzuki,
        do_ecapture, symbolic,
    )
}

/// Evaluates `eval_net` into a reusable [`NetWorkState`] without converting
/// `NetEvalResult` quantities back to molar `Y`-units.
#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_net_result_with_work(
    g: &NetGeneralInfo, w: &mut NetWorkState, rates_only: bool, just_dxdt: bool,
    x: &[f64], mut temp: f64, mut log_t: f64, rho: f64, log_rho: f64,
    abar: f64, zbar: f64, z2bar: f64, ye: f64,
    eta: f64, d_eta_dln_t: f64, d_eta_dln_rho: f64,
    rate_factors: &[f64], weak_rate_factor: f64,
    screening_mode: ScreeningMode, use_3a_fl87: bool,
    use_suzuki: bool, do_ecapture: bool, symbolic: bool,
) -> Result<NetEvalResult, NetError> {
    w.reset(g);
    eval_net_setup(
        g, w, x, &mut temp, &mut log_t, rho, abar, zbar, z2bar, ye, eta,
        d_eta_dln_t, d_eta_dln_rho, rate_factors, weak_rate_factor,
        use_suzuki, do_ecapture,
    )?;

    if g.doing_approx21 {
        return eval_net_approx21_procs(
            g, w, rates_only, just_dxdt, temp, log_t, rho, log_rho, abar,
            zbar, z2bar, ye, eta, weak_rate_factor, screening_mode,
            use_3a_fl87,
        );
    }

    screen_net(
        g, &w.y, temp, rho, log_t, log_rho,
        &w.rate_raw[..g.num_reactions],
        &w.rate_raw_dt[..g.num_reactions],
        &w.rate_raw_drho[..g.num_reactions],
        &mut w.rate_screened[..g.num_reactions],
        &mut w.rate_screened_dt[..g.num_reactions],
        &mut w.rate_screened_drho[..g.num_reactions],
        screening_mode, zbar, abar, z2bar,
    )?;

    let ni = g.num_isos;
    let nr = g.num_reactions;
    let mut res = NetEvalResult {
        eps_nuc: 0.0, d_eps_nuc_drho: 0.0, d_eps_nuc_dt: 0.0,
        d_eps_nuc_dx: vec![0.0; ni], dxdt: vec![0.0; ni],
        d_dxdt_drho: vec![0.0; ni], d_dxdt_dt: vec![0.0; ni],
        d_dxdt_dx: vec![0.0; ni * ni],
        eps_nuc_categories: [0.0; NUM_CATEGORIES],
        eps_neu_total: 0.0, rate_raw: Vec::new(), rate_screened: Vec::new(),
    };
    if rates_only {
        res.rate_raw = w.rate_raw[..nr].to_vec();
        res.rate_screened = w.rate_screened[..nr].to_vec();
        return Ok(res);
    }

    let mut stack_dydt = [[Quad::ZERO; NUM_RVS]; 128];
    let mut heap_dydt = Vec::new();
    let dydt: &mut [[Quad; NUM_RVS]] = if ni <= 128 {
        &mut stack_dydt[..ni]
    } else {
        heap_dydt.resize(ni, [Quad::ZERO; NUM_RVS]);
        &mut heap_dydt[..]
    };
    let mut eps_nuc_mev = [Quad::ZERO; NUM_RVS];
    get_derivs(
        g, w, dydt, &mut eps_nuc_mev, temp, rho, abar, zbar,
        rate_factors, use_3a_fl87, symbolic, just_dxdt,
    )?;

    res.rate_raw = w.rate_raw[..nr].to_vec();
    res.rate_screened = w.rate_screened[..nr].to_vec();
    if symbolic {
        res.d_dxdt_dx = w.d_dydt_dy[..ni * ni].to_vec();
        return Ok(res);
    }

    let nuclides = NuclideCatalog::builtin_ref();
    let mut stack_aion = [1.0_f64; 128];
    let mut heap_aion = Vec::new();
    let aion: &mut [f64] = if ni <= 128 {
        &mut stack_aion[..ni]
    } else {
        heap_aion.resize(ni, 1.0);
        &mut heap_aion[..]
    };
    for (i, a_slot) in aion.iter_mut().enumerate() {
        *a_slot = nuclide_aion(nuclides, g.chem_id[i])?;
    }
    for k in 0..NUM_CATEGORIES {
        res.eps_nuc_categories[k] = QCONV * w.eps_nuc_categories[k];
    }
    for i in 0..ni {
        res.dxdt[i] = (Quad::from_f64(aion[i]) * dydt[i][0]).to_f64();
    }
    if !just_dxdt {
        for i in 0..ni {
            let a_i = aion[i];
            res.d_eps_nuc_dx[i] = QCONV * w.d_eps_nuc_dy[i] / a_i;
            res.d_dxdt_drho[i] = (Quad::from_f64(a_i) * dydt[i][2]).to_f64();
            res.d_dxdt_dt[i] = (Quad::from_f64(a_i) * dydt[i][1]).to_f64();
            for (j, &a_j) in aion.iter().enumerate() {
                res.d_dxdt_dx[i * ni + j] = w.d_dydt_dy[i * ni + j] * a_i / a_j;
            }
        }
    }
    res.eps_nuc = (eps_nuc_mev[0] * QCONV).to_f64();
    res.d_eps_nuc_dt = (eps_nuc_mev[1] * QCONV).to_f64();
    res.d_eps_nuc_drho = (eps_nuc_mev[2] * QCONV).to_f64();
    res.eps_neu_total = w.eps_neu_total * QCONV;
    Ok(res)
}

/// Executes `f` with a thread-local [`NetWorkState`] keyed by [`NetGeneralInfo::identity_id`]
/// and sized for `g`, avoiding per-zone heap allocations and cross-network cache
/// aliasing in multi-zone sequential or parallel loops.
pub fn with_thread_local_work<R>(
    g: &NetGeneralInfo,
    f: impl FnOnce(&mut NetWorkState) -> R,
) -> R {
    thread_local! {
        static WORK: std::cell::RefCell<
            std::collections::HashMap<u64, NetWorkState>,
        > = std::cell::RefCell::new(std::collections::HashMap::new());
    }
    WORK.with(|cell| {
        let mut map = cell.borrow_mut();
        let entry = map
            .entry(g.identity_id)
            .or_insert_with(|| NetWorkState::new(g));
        if entry.y.len() != g.num_isos
            || entry.rate_raw.len() < g.num_reactions.max(120)
            || entry.wk_lambda.len() != g.num_wk_reactions
        {
            *entry = NetWorkState::new(g);
        }
        f(entry)
    })
}

/// Evaluates a nuclear reaction network in place using a reusable
/// [`NetWorkState`] with zero heap allocations.
///
/// For non-`approx21` networks, sets `NetWorkState::outputs` to `Molar`
/// (`eps_nuc_categories`, `d_eps_nuc_dy`, `d_dydt_dy`, `eps_neu_total`) and
/// fills `dydt`, while poisoning unused `direct_*` buffers with `f64::NAN`.
/// When `g.doing_approx21` is true, poisons `dydt` (`Quad::NAN`) and molar
/// buffers (`f64::NAN`) and sets `NetWorkState::outputs` to `DirectCgs`
/// (`net/private/net_eval.f90:215-238`). Returns
/// `(eps_nuc, d_eps_nuc_dt, d_eps_nuc_drho, eps_neu_total)` in
/// $\mathrm{erg\,g^{-1}\,s^{-1}}$.
#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
pub fn eval_net_with_work(
    g: &NetGeneralInfo, w: &mut NetWorkState, dydt: &mut [[Quad; NUM_RVS]],
    just_dxdt: bool, x: &[f64], mut temp: f64, mut log_t: f64,
    rho: f64, log_rho: f64, abar: f64, zbar: f64, z2bar: f64, ye: f64,
    eta: f64, d_eta_dln_t: f64, d_eta_dln_rho: f64,
    rate_factors: &[f64], weak_rate_factor: f64,
    screening_mode: ScreeningMode, use_3a_fl87: bool,
    use_suzuki: bool, do_ecapture: bool,
) -> Result<(f64, f64, f64, f64), NetError> {
    w.reset(g);
    let ni = g.num_isos;
    if g.doing_approx21 {
        dydt[..ni].fill([Quad::NAN; NUM_RVS]);
    } else {
        dydt[..ni].fill([Quad::ZERO; NUM_RVS]);
    }
    eval_net_setup(
        g, w, x, &mut temp, &mut log_t, rho, abar, zbar, z2bar, ye, eta,
        d_eta_dln_t, d_eta_dln_rho, rate_factors, weak_rate_factor,
        use_suzuki, do_ecapture,
    )?;
    if g.doing_approx21 {
        // Export directly in CGS / mass-fraction units (`net_eval.f90:232-237,
        // 308-338, 428-438`) without `aion` / `QCONV` round-trips.
        return eval_net_approx21_into(
            g,
            w,
            false,
            just_dxdt,
            temp,
            log_t,
            rho,
            log_rho,
            abar,
            zbar,
            z2bar,
            ye,
            eta,
            weak_rate_factor,
            screening_mode,
            use_3a_fl87,
        );
    }
    screen_net(
        g, &w.y, temp, rho, log_t, log_rho,
        &w.rate_raw[..g.num_reactions],
        &w.rate_raw_dt[..g.num_reactions],
        &w.rate_raw_drho[..g.num_reactions],
        &mut w.rate_screened[..g.num_reactions],
        &mut w.rate_screened_dt[..g.num_reactions],
        &mut w.rate_screened_drho[..g.num_reactions],
        screening_mode, zbar, abar, z2bar,
    )?;
    let mut eps_nuc_mev = [Quad::ZERO; NUM_RVS];
    get_derivs(
        g, w, &mut dydt[..ni], &mut eps_nuc_mev, temp, rho, abar, zbar,
        rate_factors, use_3a_fl87, false, just_dxdt,
    )?;
    let eps_nuc = (eps_nuc_mev[0] * QCONV).to_f64();
    let d_eps_nuc_dt = (eps_nuc_mev[1] * QCONV).to_f64();
    let d_eps_nuc_drho = (eps_nuc_mev[2] * QCONV).to_f64();
    let eps_neu_total = w.eps_neu_total * QCONV;
    Ok((eps_nuc, d_eps_nuc_dt, d_eps_nuc_drho, eps_neu_total))
}

/// Evaluates molar RHS `dydt`, `eps_nuc_mev`, and `w.d_dydt_dy` across all
/// reactions in `g` (`get_derivs` in `net/private/net_derivs.f90`).
#[allow(clippy::too_many_arguments)]
pub fn get_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[Quad; NUM_RVS]],
    eps_nuc_mev: &mut [Quad; NUM_RVS],
    temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
    rate_factors: &[f64],
    use_3a_fl87: bool,
    symbolic: bool,
    just_dydt: bool,
) -> Result<(), NetError> {
    let nr = g.num_reactions;
    for i in 0..nr {
        update_special_rates(
            g,
            w,
            dydt,
            eps_nuc_mev,
            i,
            temp,
            den,
            abar,
            zbar,
            rate_factors,
            use_3a_fl87,
            symbolic,
            just_dydt,
        );
    }

    let mut i = 0;
    while i < nr {
        match g.reaction_reaclib_kind[i] {
            ReaclibKind::Other => {
                get1_derivs(
                    g,
                    w,
                    dydt,
                    eps_nuc_mev,
                    i,
                    temp,
                    den,
                    use_3a_fl87,
                    symbolic,
                    just_dydt,
                )?;
                i += 1;
            }
            ReaclibKind::Ng | ReaclibKind::Pg | ReaclibKind::Ag => {
                get_basic_2_to_1_derivs(
                    g,
                    w,
                    dydt,
                    eps_nuc_mev,
                    i,
                    symbolic,
                    just_dydt,
                );
                i += 2;
            }
            ReaclibKind::Pn | ReaclibKind::Ap | ReaclibKind::An => {
                get_basic_2_to_2_derivs(
                    g,
                    w,
                    dydt,
                    eps_nuc_mev,
                    i,
                    symbolic,
                    just_dydt,
                );
                i += 2;
            }
            ReaclibKind::GeneralOneOne => {
                get_general_1_to_1_derivs(
                    g,
                    w,
                    dydt,
                    eps_nuc_mev,
                    i,
                    symbolic,
                    just_dydt,
                );
                i += 2;
            }
            ReaclibKind::GeneralTwoOne => {
                get_general_2_to_1_derivs(
                    g,
                    w,
                    dydt,
                    eps_nuc_mev,
                    i,
                    symbolic,
                    just_dydt,
                );
                i += 2;
            }
            ReaclibKind::GeneralTwoTwo => {
                get_general_2_to_2_derivs(
                    g,
                    w,
                    dydt,
                    eps_nuc_mev,
                    i,
                    symbolic,
                    just_dydt,
                );
                i += 2;
            }
        }
    }
    w.set_eval_branch(false);
    Ok(())
}

/// `get1_derivs` (`net/private/net_derivs.f90:1073-1545`) for one
/// reaction of the `other_kind` path, with upstream's slot handling and
/// error stops (#661).
///
/// # Errors
///
/// Returns [`NetError::IntegrationError`] naming the reaction where upstream
/// calls `mesa_error`: `itab(outK) = 0` (`:1185-1212`),
/// `num_reaction_outputs >= 4` (`:1215-1218`), `itab(in1) = 0` for one input
/// (`:1264-1268`), an invalid coefficient pair (`:1326-1329`),
/// `itab(in3) = 0` (`:1335-1339`), an invalid input count (`:1343-1346`), an
/// output count the dispatch does not allow (`:1364-1477`), and
/// `weak_id > num_wk_reactions` (`:1489-1496`). A two- or three-input
/// reaction with `itab(in1)` or `itab(in2)` zero would read `n% y(0)`,
/// outside the array, which has no defined value (DECISIONS #43).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn get1_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[Quad; NUM_RVS]],
    eps_nuc_mev: &mut [Quad; NUM_RVS],
    i: usize,
    temp: f64,
    den: f64,
    use_3a_fl87: bool,
    symbolic: bool,
    just_dydt: bool,
) -> Result<(), NetError> {
    let rxn = &g.reactions[i];
    let fail = |what: &str| {
        Err(NetError::IntegrationError(format!(
            "get1_derivs: {} {what} (net/private/net_derivs.f90)",
            rxn.name
        )))
    };
    if rxn.out_num[0] == 0 {
        w.raw_rate[i] = 0.0;
        w.screened_rate[i] = 0.0;
        w.eps_nuc_rate[i] = 0.0;
        w.eps_neu_rate[i] = 0.0;
        return Ok(());
    }
    if (rxn.name == "r_he4_he4_he4_to_c12" && use_3a_fl87)
        || rxn.name == "rn14ag_lite"
    {
        return Ok(());
    }

    let n_in = rxn.num_in;
    let n_out = rxn.num_out;

    let prot = g.net_iso[IPROTON];
    let h1 = g.net_iso[IH1];

    // Outputs (`:1180-1218`): each used slot must be in the net, and at
    // most three outputs.
    let mut i_out = [0usize; 3];
    let mut c_out = [0.0; 3];
    for k in 0..n_out.min(3) {
        c_out[k] = f64::from(rxn.out_num[k]);
        i_out[k] = g.net_iso[rxn.out_iso[k] as usize];
        if i_out[k] == 0 {
            return fail(&format!("itab(out{}) = 0", k + 1));
        }
    }
    if n_out >= 4 {
        return fail("num_reaction_outputs >= 4");
    }

    // Inputs (`:1220-1236`).
    let mut i_in = [0usize; 3];
    let mut c_in = [0.0; 3];
    let mut in1 = 0usize;
    if (1..=3).contains(&n_in) {
        for k in 0..n_in {
            c_in[k] = f64::from(rxn.in_num[k]);
            i_in[k] = g.net_iso[rxn.in_iso[k] as usize];
        }
        in1 = rxn.in_iso[0] as usize;
    }

    // `switch_to_prot` (`:1238-1259`): the first matching slot on each side
    // only, testing every slot index (unused ones are 0) against `h1`.
    let switch_to_prot =
        prot != 0 && g.reaction_max_z[i] >= MIN_Z_FOR_SWITCH_TO_PROT;
    if switch_to_prot {
        if i_in[0] == h1 {
            in1 = IPROTON;
            i_in[0] = prot;
        } else if i_in[1] == h1 {
            i_in[1] = prot;
        } else if i_in[2] == h1 {
            i_in[2] = prot;
        }
        if i_out[0] == h1 {
            i_out[0] = prot;
        } else if i_out[1] == h1 {
            i_out[1] = prot;
        } else if i_out[2] == h1 {
            i_out[2] = prot;
        }
    }

    let r;
    let mut idr = [0usize; 3];
    let mut dr = [0.0; 3];

    if n_in == 1 {
        let i1 = i_in[0];
        if i1 == 0 {
            return fail("itab(in1) = 0");
        }
        let y1 = w.y[i1 - 1];
        let cin1 = rxn.in_num[0];
        if cin1 == 1 {
            r = y1;
            idr[0] = i1;
            dr[0] = 1.0;
        } else if cin1 == 3 && in1 != IH1 {
            r = (1.0 / 6.0) * y1 * y1 * y1;
            idr[0] = i1;
            dr[0] = 0.5 * y1 * y1;
        } else {
            r = 0.5 * y1 * y1;
            idr[0] = i1;
            dr[0] = y1;
        }
    } else if n_in == 2 || n_in == 3 {
        let i1 = i_in[0];
        let i2 = i_in[1];
        if i1 == 0 || i2 == 0 {
            // Upstream reads `n% y(0)` here (DECISIONS #43, #661).
            return fail("itab(in1) or itab(in2) = 0 reads n% y(0)");
        }
        let y1 = w.y[i1 - 1];
        let y2 = w.y[i2 - 1];
        let cin1 = rxn.in_num[0];
        let cin2 = rxn.in_num[1];
        if rxn.rho_exp == 0 {
            r = y1;
            idr[0] = i1;
            dr[0] = 1.0;
            idr[1] = i2;
            dr[1] = 0.0;
        } else if (cin1 == 1 && cin2 == 1) || rxn.rho_exp == 1 {
            r = y1 * y2;
            dr[0] = y1;
            idr[0] = i2;
            dr[1] = y2;
            idr[1] = i1;
        } else if cin1 == 2 && cin2 == 1 {
            r = 0.5 * y1 * y1 * y2;
            dr[0] = 0.5 * y1 * y1;
            idr[0] = i2;
            dr[1] = y1 * y2;
            idr[1] = i1;
        } else if cin1 == 1 && cin2 == 2 {
            r = y1 * 0.5 * y2 * y2;
            dr[0] = y1 * y2;
            idr[0] = i2;
            dr[1] = 0.5 * y2 * y2;
            idr[1] = i1;
        } else if cin1 == 2 && cin2 == 2 {
            r = 0.5 * y1 * y1 * 0.5 * y2 * y2;
            dr[0] = 0.5 * y1 * y1 * y2;
            idr[0] = i2;
            dr[1] = y1 * 0.5 * y2 * y2;
            idr[1] = i1;
        } else {
            return fail("invalid coefficient");
        }
        if n_in == 3 {
            dr[2] = 0.0;
            idr[2] = i_in[2];
            if i_in[2] == 0 {
                return fail("itab(in3) = 0");
            }
        }
    } else {
        return fail("invalid specification");
    }

    // Output counts the dispatch allows (`:1364-1476`).
    let outputs_ok = match n_in {
        1 | 2 => (1..=3).contains(&n_out),
        _ => (1..=2).contains(&n_out),
    };
    if !outputs_ok {
        return fail(&format!(
            "too many reaction_outputs for num_reaction_inputs == {n_in}"
        ));
    }

    if n_in == 1 && n_out == 1 {
        let weak_id = g.weak_reaction_index[i];
        if weak_id > g.num_wk_reactions {
            return fail(&format!(
                "weak_id {weak_id} > num_wk_reactions {}",
                g.num_wk_reactions
            ));
        }
        if weak_id > 0 && g.weaklib_ids[weak_id - 1] > 0 {
            let wk = weak_id - 1;
            w.rate_screened[i] = w.wk_lambda[wk];
            w.rate_screened_dt[i] = w.wk_dlambda_dln_t[wk] / temp;
            w.rate_screened_drho[i] = w.wk_dlambda_dln_rho[wk] / den;
            do_in_out_neu(
                g,
                w,
                dydt,
                eps_nuc_mev,
                i,
                r,
                n_in,
                i_in,
                c_in,
                n_out,
                i_out,
                c_out,
                idr,
                dr,
                w.wk_q[wk],
                w.wk_qneu[wk],
                w.wk_dqneu_dln_t[wk] / temp,
                w.wk_dqneu_dln_rho[wk] / den,
                symbolic,
                just_dydt,
            );
            return Ok(());
        }
    }

    do_in_out(
        g,
        w,
        dydt,
        eps_nuc_mev,
        i,
        r,
        n_in,
        i_in,
        c_in,
        n_out,
        i_out,
        c_out,
        idr,
        dr,
        symbolic,
        just_dydt,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ParsedNetDef;

    #[test]
    fn test_thread_local_work_isolates_same_size_distinct_networks() {
        let p1 = ParsedNetDef::parse_from_name("basic.net").expect("basic.net");
        let mut p2 = p1.clone();
        p2.net_filename = String::from("basic_distinct_variant.net");
        let g1 = NetGeneralInfo::from_parsed(p1).expect("g1");
        let g2 = NetGeneralInfo::from_parsed(p2).expect("g2");

        assert_eq!(g1.num_isos, g2.num_isos);
        assert_eq!(g1.num_reactions, g2.num_reactions);
        assert_eq!(g1.num_wk_reactions, g2.num_wk_reactions);
        assert_ne!(g1.identity_id, g2.identity_id);

        with_thread_local_work(&g1, |w| {
            w.y[0] = 42.25;
        });
        with_thread_local_work(&g2, |w| {
            assert_eq!(w.y[0], 0.0);
            w.y[0] = 99.5;
        });
        with_thread_local_work(&g1, |w| {
            assert_eq!(w.y[0], 42.25);
        });
        with_thread_local_work(&g2, |w| {
            assert_eq!(w.y[0], 99.5);
        });
    }

    /// Runs `get1_derivs` on reaction `i` of `g` with unit abundances.
    fn run_get1(g: &NetGeneralInfo, i: usize) -> Result<(), NetError> {
        let mut w = NetWorkState::new(g);
        w.y.fill(0.1);
        let mut dydt = vec![[Quad::default(); NUM_RVS]; g.num_isos];
        let mut eps = [Quad::default(); NUM_RVS];
        get1_derivs(
            g, &mut w, &mut dydt, &mut eps, i, 1.0e8, 1.0e2, false, false,
            false,
        )
    }

    /// The first two-input, one-output `other` reaction of `basic.net`.
    fn basic_two_in_one_out() -> (NetGeneralInfo, usize) {
        let g = NetGeneralInfo::from_name("basic.net").expect("basic.net");
        let i = (0..g.num_reactions)
            .find(|&i| {
                let r = &g.reactions[i];
                g.reaction_reaclib_kind[i] == ReaclibKind::Other
                    && r.num_in == 2
                    && r.num_out == 1
                    && r.out_num[0] != 0
                    && r.name != "rn14ag_lite"
            })
            .expect("a 2 -> 1 reaction");
        (g, i)
    }

    /// Each upstream `mesa_error` stop in `get1_derivs`
    /// (`net/private/net_derivs.f90:1185-1496`) is an explicit error naming
    /// the reaction, not a clamp, a silent zero or an index panic (#661).
    #[test]
    fn get1_derivs_upstream_stops_are_errors() {
        let (g, i) = basic_two_in_one_out();
        assert!(run_get1(&g, i).is_ok());
        let msg =
            |g: &NetGeneralInfo| run_get1(g, i).expect_err("error").to_string();

        let mut bad = g.clone();
        bad.reactions[i].num_out = 4;
        bad.reactions[i].out_iso = [bad.reactions[i].out_iso[0]; 3];
        bad.reactions[i].out_num = [1; 3];
        assert!(msg(&bad).contains("num_reaction_outputs >= 4"));

        let mut bad = g.clone();
        bad.reactions[i].num_in = 4;
        assert!(msg(&bad).contains("invalid specification"));

        let mut bad = g.clone();
        bad.reactions[i].in_num = [3, 3, 0];
        bad.reactions[i].rho_exp = 2;
        assert!(msg(&bad).contains("invalid coefficient"));

        let mut bad = g.clone();
        bad.reactions[i].num_in = 3;
        bad.reactions[i].in_num[2] = 1;
        bad.reactions[i].in_iso[2] = 0;
        assert!(msg(&bad).contains("itab(in3) = 0"));

        let mut bad = g.clone();
        bad.reactions[i].num_out = 3;
        bad.reactions[i].out_num = [1; 3];
        bad.reactions[i].out_iso[1] = bad.reactions[i].out_iso[0];
        bad.reactions[i].out_iso[2] = 0;
        assert!(msg(&bad).contains("itab(out3) = 0"));
    }
}
