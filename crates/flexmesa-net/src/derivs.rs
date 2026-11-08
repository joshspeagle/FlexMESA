// SPDX-License-Identifier: LGPL-3.0-or-later
//! General reaction-network RHS, analytical Jacobians, and energy generation
//! (`net_derivs.f90` and `net_eval.f90`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::QCONV;
use flexmesa_rates::ScreeningMode;

use crate::approx21_eval::eval_net_approx21_procs;
use crate::derivs_paired::{
    get_basic_2_to_1_derivs, get_basic_2_to_2_derivs,
    get_general_1_to_1_derivs, get_general_2_to_1_derivs,
    get_general_2_to_2_derivs,
};
use crate::derivs_support::{
    do_in_out, do_in_out_neu, get_weaklib_rates, update_special_rates,
    NetWorkState,
};
use crate::f64x2::F64x2;
use crate::screen::screen_net;
use crate::types::{
    NetError, NetEvalResult, NetGeneralInfo, ReaclibKind, IH1, IPROTON,
    NUM_CATEGORIES, NUM_RVS, RATTAB_TEMP_LO, RATTAB_TLO,
};

const MIN_Z_FOR_SWITCH_TO_PROT: i32 = 12;

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
        let a = nuclides.get_by_id(cid).unwrap().z_plus_n as f64;
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
    mut temp: f64, mut log_t: f64, rho: f64, log_rho: f64,
    abar: f64, zbar: f64, z2bar: f64, ye: f64,
    eta: f64, d_eta_dln_t: f64, d_eta_dln_rho: f64,
    rate_factors: &[f64], weak_rate_factor: f64,
    screening_mode: ScreeningMode, use_3a_fl87: bool,
    use_suzuki: bool, do_ecapture: bool, symbolic: bool,
) -> Result<NetEvalResult, NetError> {
    let mut w = NetWorkState::new(g);
    eval_net_setup(
        g, &mut w, x, &mut temp, &mut log_t, rho, abar, zbar, z2bar, ye, eta,
        d_eta_dln_t, d_eta_dln_rho, rate_factors, weak_rate_factor,
        use_suzuki, do_ecapture,
    )?;

    if g.doing_approx21 {
        return eval_net_approx21_procs(
            g, &mut w, rates_only, just_dxdt, temp, log_t, rho, log_rho, abar,
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

    let mut stack_dydt = [[F64x2::ZERO; NUM_RVS]; 32];
    let mut heap_dydt = Vec::new();
    let dydt: &mut [[F64x2; NUM_RVS]] = if ni <= 32 {
        &mut stack_dydt[..ni]
    } else {
        heap_dydt.resize(ni, [F64x2::ZERO; NUM_RVS]);
        &mut heap_dydt[..]
    };
    let mut eps_nuc_mev = [F64x2::ZERO; NUM_RVS];
    get_derivs(
        g, &mut w, dydt, &mut eps_nuc_mev, temp, rho, abar, zbar,
        rate_factors, use_3a_fl87, symbolic, just_dxdt,
    )?;

    res.rate_raw = w.rate_raw[..nr].to_vec();
    res.rate_screened = w.rate_screened[..nr].to_vec();
    if symbolic {
        res.d_dxdt_dx = w.d_dydt_dy;
        return Ok(res);
    }

    let nuclides = NuclideCatalog::builtin_ref();
    for k in 0..NUM_CATEGORIES {
        res.eps_nuc_categories[k] = QCONV * w.eps_nuc_categories[k];
    }
    for i in 0..ni {
        let a_i = nuclides.get_by_id(g.chem_id[i]).unwrap().z_plus_n as f64;
        res.dxdt[i] = a_i * dydt[i][0].to_f64();
    }
    if !just_dxdt {
        for i in 0..ni {
            let a_i = nuclides.get_by_id(g.chem_id[i]).unwrap().z_plus_n as f64;
            res.d_eps_nuc_dx[i] = QCONV * w.d_eps_nuc_dy[i] / a_i;
            res.d_dxdt_drho[i] = a_i * dydt[i][2].to_f64();
            res.d_dxdt_dt[i] = a_i * dydt[i][1].to_f64();
            for j in 0..ni {
                let a_j =
                    nuclides.get_by_id(g.chem_id[j]).unwrap().z_plus_n as f64;
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

/// Executes `f` with a thread-local [`NetWorkState`] sized for `g`, avoiding
/// per-zone heap allocations in multi-zone sequential or parallel loops.
pub fn with_thread_local_work<R>(
    g: &NetGeneralInfo,
    f: impl FnOnce(&mut NetWorkState) -> R,
) -> R {
    thread_local! {
        static WORK: std::cell::RefCell<Option<NetWorkState>> =
            const { std::cell::RefCell::new(None) };
    }
    WORK.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let needs_init = match borrow.as_ref() {
            None => true,
            Some(w) => {
                w.y.len() < g.num_isos
                    || w.rate_raw.len() < g.num_reactions
                    || w.wk_lambda.len() < g.num_wk_reactions
            }
        };
        if needs_init {
            *borrow = Some(NetWorkState::new(g));
        }
        f(borrow.as_mut().unwrap())
    })
}

/// Evaluates a nuclear reaction network in place using a reusable
/// [`NetWorkState`] and `dydt` slice with zero heap allocations.
///
/// Populates `w` and `dydt`, and returns
/// `(eps_nuc, d_eps_nuc_dt, d_eps_nuc_drho, eps_neu_total)` in
/// $\mathrm{erg\,g^{-1}\,s^{-1}}$.
#[rustfmt::skip]
#[allow(clippy::too_many_arguments)]
pub fn eval_net_with_work(
    g: &NetGeneralInfo, w: &mut NetWorkState, dydt: &mut [[F64x2; NUM_RVS]],
    just_dxdt: bool, x: &[f64], mut temp: f64, mut log_t: f64,
    rho: f64, log_rho: f64, abar: f64, zbar: f64, z2bar: f64, ye: f64,
    eta: f64, d_eta_dln_t: f64, d_eta_dln_rho: f64,
    rate_factors: &[f64], weak_rate_factor: f64,
    screening_mode: ScreeningMode, use_3a_fl87: bool,
    use_suzuki: bool, do_ecapture: bool,
) -> Result<(f64, f64, f64, f64), NetError> {
    w.reset(g);
    let ni = g.num_isos;
    dydt[..ni].fill([F64x2::ZERO; NUM_RVS]);
    eval_net_setup(
        g, w, x, &mut temp, &mut log_t, rho, abar, zbar, z2bar, ye, eta,
        d_eta_dln_t, d_eta_dln_rho, rate_factors, weak_rate_factor,
        use_suzuki, do_ecapture,
    )?;
    if g.doing_approx21 {
        let res = eval_net_approx21_procs(
            g, w, false, just_dxdt, temp, log_t, rho, log_rho, abar, zbar,
            z2bar, ye, eta, weak_rate_factor, screening_mode, use_3a_fl87,
        )?;
        let nuclides = NuclideCatalog::builtin_ref();
        for i in 0..ni {
            let a_i = nuclides.get_by_id(g.chem_id[i]).unwrap().z_plus_n as f64;
            dydt[i][0] = F64x2::from_f64(res.dxdt[i] / a_i);
            if !just_dxdt {
                dydt[i][1] = F64x2::from_f64(res.d_dxdt_dt[i] / a_i);
                dydt[i][2] = F64x2::from_f64(res.d_dxdt_drho[i] / a_i);
            }
        }
        return Ok((
            res.eps_nuc, res.d_eps_nuc_dt, res.d_eps_nuc_drho,
            res.eps_neu_total,
        ));
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
    let mut eps_nuc_mev = [F64x2::ZERO; NUM_RVS];
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
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
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
                );
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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn get1_derivs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    dydt: &mut [[F64x2; NUM_RVS]],
    eps_nuc_mev: &mut [F64x2; NUM_RVS],
    i: usize,
    temp: f64,
    den: f64,
    use_3a_fl87: bool,
    symbolic: bool,
    just_dydt: bool,
) {
    let rxn = &g.reactions[i];
    if rxn.out_num[0] == 0 {
        w.raw_rate[i] = 0.0;
        w.screened_rate[i] = 0.0;
        w.eps_nuc_rate[i] = 0.0;
        w.eps_neu_rate[i] = 0.0;
        return;
    }
    if (rxn.name == "r_he4_he4_he4_to_c12" && use_3a_fl87)
        || rxn.name == "rn14ag_lite"
    {
        return;
    }

    let n_in = rxn.num_in.min(3);
    let n_out = rxn.num_out.min(3);

    let prot = g.net_iso[IPROTON];
    let h1 = g.net_iso[IH1];
    let switch_to_prot =
        prot != 0 && g.reaction_max_z[i] >= MIN_Z_FOR_SWITCH_TO_PROT;

    let mut i_in = [0usize; 3];
    let mut c_in = [0.0; 3];
    let mut in_cid = [0usize; 3];
    for k in 0..n_in {
        c_in[k] = f64::from(rxn.in_num[k]);
        in_cid[k] = rxn.in_iso[k] as usize;
        i_in[k] = g.net_iso[in_cid[k]];
        if switch_to_prot && i_in[k] == h1 {
            in_cid[k] = IPROTON;
            i_in[k] = prot;
        }
    }

    let mut i_out = [0usize; 3];
    let mut c_out = [0.0; 3];
    for k in 0..n_out {
        c_out[k] = f64::from(rxn.out_num[k]);
        let cid = rxn.out_iso[k] as usize;
        i_out[k] = g.net_iso[cid];
        if switch_to_prot && i_out[k] == h1 {
            i_out[k] = prot;
        }
    }

    let mut r = 0.0;
    let mut idr = [0usize; 3];
    let mut dr = [0.0; 3];

    if n_in == 1 {
        let i1 = i_in[0];
        let y1 = w.y[i1 - 1];
        let cin1 = rxn.in_num[0];
        if cin1 == 1 {
            r = y1;
            idr[0] = i1;
            dr[0] = 1.0;
        } else if cin1 == 3 && in_cid[0] != IH1 {
            r = (1.0 / 6.0) * y1 * y1 * y1;
            idr[0] = i1;
            dr[0] = 0.5 * y1 * y1;
        } else {
            r = 0.5 * y1 * y1;
            idr[0] = i1;
            dr[0] = y1;
        }
    } else {
        let i1 = i_in[0];
        let i2 = i_in[1];
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
        }
        if n_in == 3 {
            idr[2] = i_in[2];
            dr[2] = 0.0;
        }
    }

    if n_in == 1 && n_out == 1 {
        let weak_id = g.weak_reaction_index[i];
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
            return;
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
}
