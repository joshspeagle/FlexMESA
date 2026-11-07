// SPDX-License-Identifier: LGPL-3.0-or-later
//! Temperature/density sensitivities (`approx21_dfdt_dfdrho`) and full
//! `approx21` evaluation driver (`eval_net_approx21_procs`).

use flexmesa_chem::NuclideCatalog;
use flexmesa_const::{AVO, CLIGHT, QCONV};
use flexmesa_rates::ScreeningMode;

use crate::approx21::{
    approx21_pa_pg_fractions, approx21_weak_rates, Approx21Indices,
};
use crate::approx21_dfdy::approx21_dfdy;
use crate::approx21_dydt::approx21_dydt;
use crate::approx21_eps::{
    approx21_d_epsneu_dy, approx21_eps_info, approx21_eval_ppii_fraction,
};
use crate::approx21_special::approx21_special_reactions;
use crate::derivs_support::NetWorkState;
use crate::screen::screen_net;
use crate::types::{
    NetError, NetEvalResult, NetGeneralInfo, APPROX21_CO56_NUM_REACTIONS,
    APPROX21_NUM_REACTIONS, NUM_CATEGORIES,
};

/// Evaluates `approx21_dfdT_dfdRho` (`net/private/net_approx21.f90` lines 2655–2695).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub fn approx21_dfdt_dfdrho(
    idx: &Approx21Indices,
    species: usize,
    y: &[f64],
    ratdum: &[f64],
    dratdumdt: &[f64],
    dratdumdd: &[f64],
    mion: &[f64],
    fe56ec_fake_factor_in: f64,
    min_t_for_fe56ec_fake_factor: f64,
    fe56ec_n_neut: i32,
    temp: f64,
    dfdy: &[f64],
    dfdt: &mut [f64],
    dfdrho: &mut [f64],
    d_epstotal_dy: &mut [f64],
) {
    approx21_dydt(
        idx,
        y,
        ratdum,
        dratdumdt,
        true,
        fe56ec_fake_factor_in,
        min_t_for_fe56ec_fake_factor,
        fe56ec_n_neut,
        temp,
        dfdt,
    );
    approx21_dydt(
        idx,
        y,
        ratdum,
        dratdumdd,
        true,
        fe56ec_fake_factor_in,
        min_t_for_fe56ec_fake_factor,
        fe56ec_n_neut,
        temp,
        dfdrho,
    );
    let enuc_conv2 = -AVO * CLIGHT * CLIGHT;
    for j in 0..species {
        let mut xx = 0.0_f64;
        for i in 0..species {
            xx += dfdy[i * species + j] * mion[i];
        }
        d_epstotal_dy[j] = xx * enuc_conv2;
    }
}

/// Evaluates the full `eval_net_approx21_procs` pipeline (`net/private/net_eval.f90`
/// lines 302–440 and 621–672).
#[allow(clippy::too_many_arguments)]
pub fn eval_net_approx21_procs(
    g: &NetGeneralInfo,
    w: &mut NetWorkState,
    rates_only: bool,
    just_dxdt: bool,
    temp: f64,
    log_t: f64,
    rho: f64,
    log_rho: f64,
    abar: f64,
    zbar: f64,
    z2bar: f64,
    ye: f64,
    eta: f64,
    weak_rate_factor: f64,
    screening_mode: ScreeningMode,
    use_3a_fl87: bool,
) -> Result<NetEvalResult, NetError> {
    let idx = Approx21Indices::from_net(g);
    let m = g.num_reactions;
    let total_r = if g.add_co56_to_approx21 {
        APPROX21_CO56_NUM_REACTIONS
    } else {
        APPROX21_NUM_REACTIONS
    };
    if w.rate_raw.len() < total_r {
        w.rate_raw.resize(total_r, 0.0);
        w.rate_raw_dt.resize(total_r, 0.0);
        w.rate_raw_drho.resize(total_r, 0.0);
        w.rate_screened.resize(total_r, 0.0);
        w.rate_screened_dt.resize(total_r, 0.0);
        w.rate_screened_drho.resize(total_r, 0.0);
    }

    approx21_pa_pg_fractions(
        &idx,
        &mut w.rate_raw,
        &mut w.rate_raw_dt,
        &mut w.rate_raw_drho,
    );
    approx21_weak_rates(
        &idx,
        &mut w.rate_raw,
        &mut w.rate_raw_dt,
        &mut w.rate_raw_drho,
        temp,
        rho,
        ye,
        eta,
        weak_rate_factor,
    );

    screen_net(
        g,
        &w.y,
        temp,
        rho,
        log_t,
        log_rho,
        &w.rate_raw[..m],
        &w.rate_raw_dt[..m],
        &w.rate_raw_drho[..m],
        &mut w.rate_screened[..m],
        &mut w.rate_screened_dt[..m],
        &mut w.rate_screened_drho[..m],
        screening_mode,
        zbar,
        abar,
        z2bar,
    )?;

    for i in m..total_r {
        w.rate_screened[i] = w.rate_raw[i];
        w.rate_screened_dt[i] = w.rate_raw_dt[i];
        w.rate_screened_drho[i] = w.rate_raw_drho[i];
    }

    let species = g.num_isos;
    let mut res = NetEvalResult {
        eps_nuc: 0.0,
        d_eps_nuc_drho: 0.0,
        d_eps_nuc_dt: 0.0,
        d_eps_nuc_dx: vec![0.0; species],
        dxdt: vec![0.0; species],
        d_dxdt_drho: vec![0.0; species],
        d_dxdt_dt: vec![0.0; species],
        d_dxdt_dx: vec![0.0; species * species],
        eps_nuc_categories: [0.0; NUM_CATEGORIES],
        eps_neu_total: 0.0,
        rate_raw: w.rate_raw[..total_r].to_vec(),
        rate_screened: w.rate_screened[..total_r].to_vec(),
    };
    if rates_only {
        return Ok(res);
    }

    let mut dratdumdy1 = vec![0.0_f64; total_r];
    let mut dratdumdy2 = vec![0.0_f64; total_r];
    let conv_eps_3a = QCONV * g.reactions[idx.ir3a].std_q;
    approx21_special_reactions(
        &idx,
        temp,
        rho,
        abar,
        zbar,
        &w.y,
        use_3a_fl87,
        conv_eps_3a,
        &mut w.rate_screened,
        &mut w.rate_screened_dt,
        &mut w.rate_screened_drho,
        &mut dratdumdy1,
        &mut dratdumdy2,
    );
    res.rate_screened = w.rate_screened[..total_r].to_vec();

    let mut dydt1 = vec![0.0_f64; species];
    approx21_dydt(
        &idx,
        &w.y,
        &w.rate_screened,
        &w.rate_screened,
        false,
        g.fe56ec_fake_factor,
        g.min_t_for_fe56ec_fake_factor,
        g.fe56ec_n_neut,
        temp,
        &mut dydt1,
    );

    let f_ii = approx21_eval_ppii_fraction(&idx, &w.y, &w.rate_screened);
    let (eps_total, eps_neu_total) = approx21_eps_info(
        g,
        &idx,
        &w.y,
        &dydt1,
        &w.rate_screened,
        f_ii,
        temp,
        true,
        &mut res.eps_nuc_categories,
    );
    res.eps_neu_total = eps_neu_total;
    res.eps_nuc = eps_total - eps_neu_total;

    let nuclides = NuclideCatalog::builtin_ref();
    for i in 0..species {
        let ai = nuclides.get_by_id(g.chem_id[i]).unwrap().z_plus_n as f64;
        res.dxdt[i] = dydt1[i] * ai;
    }
    if just_dxdt {
        return Ok(res);
    }

    let mut dfdy = vec![0.0_f64; species * species];
    approx21_dfdy(
        &idx,
        species,
        &w.y,
        &w.rate_screened,
        &dratdumdy1,
        &dratdumdy2,
        g.fe56ec_fake_factor,
        g.min_t_for_fe56ec_fake_factor,
        g.fe56ec_n_neut,
        temp,
        &mut dfdy,
    );

    let mut dfdt = vec![0.0_f64; species];
    let mut dfdrho = vec![0.0_f64; species];
    let mut d_epstotal_dy = vec![0.0_f64; species];
    approx21_dfdt_dfdrho(
        &idx,
        species,
        &w.y,
        &w.rate_screened,
        &w.rate_screened_dt,
        &w.rate_screened_drho,
        &g.mion,
        g.fe56ec_fake_factor,
        g.min_t_for_fe56ec_fake_factor,
        g.fe56ec_n_neut,
        temp,
        &dfdy,
        &mut dfdt,
        &mut dfdrho,
        &mut d_epstotal_dy,
    );

    let mut dummy_cats = [0.0_f64; NUM_CATEGORIES];
    let (deps_total_dt, deps_neu_dt) = approx21_eps_info(
        g,
        &idx,
        &w.y,
        &dfdt,
        &w.rate_screened_dt,
        f_ii,
        temp,
        false,
        &mut dummy_cats,
    );
    res.d_eps_nuc_dt = deps_total_dt - deps_neu_dt;

    let (deps_total_drho, deps_neu_drho) = approx21_eps_info(
        g,
        &idx,
        &w.y,
        &dfdrho,
        &w.rate_screened_drho,
        f_ii,
        temp,
        false,
        &mut dummy_cats,
    );
    res.d_eps_nuc_drho = deps_total_drho - deps_neu_drho;

    let mut d_epsneu_dy = vec![0.0_f64; species];
    approx21_d_epsneu_dy(&idx, &w.y, &w.rate_screened, &mut d_epsneu_dy);

    for i in 0..species {
        let ai = nuclides.get_by_id(g.chem_id[i]).unwrap().z_plus_n as f64;
        res.d_eps_nuc_dx[i] = (d_epstotal_dy[i] - d_epsneu_dy[i]) / ai;
        res.d_dxdt_drho[i] = dfdrho[i] * ai;
        res.d_dxdt_dt[i] = dfdt[i] * ai;
        for j in 0..species {
            let aj = nuclides.get_by_id(g.chem_id[j]).unwrap().z_plus_n as f64;
            res.d_dxdt_dx[i * species + j] = dfdy[i * species + j] * ai / aj;
        }
    }
    Ok(res)
}
