// SPDX-License-Identifier: GPL-3.0-or-later
//! Standalone IEEE-754 hex parity test for `#537` (`hb_2M` model 64 zone 1810
//! `d_epsnuc_dlnd` / `d_eps_nuc_drho` on `pp_and_cno_extras.net`).
//!
//! Uses only hardcoded IEEE-754 hex constants so no runtime paths or
//! external environment variables are needed. On mismatch, prints EOS
//! diagnostics, ULP deltas, and top reaction contributors to `d_epsnuc_dlnd`.

#![forbid(unsafe_code)]

use std::fmt::Write as _;

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{AVO, CRAD, LN10, QCONV};
use flexmesa_eos::{
    eos_dt_get_from_weights, EosConfig, EosEvalOutput, HelmTable, I_CHI_RHO,
    I_CHI_T, I_CP, I_ETA, I_GRAD_AD, I_LN_PGAS,
};
use flexmesa_math::{exp, math_init, with_numerics_mode, NumericsMode, Quad};
use flexmesa_net::derivs::{eval_net_with_work, with_thread_local_work};
use flexmesa_net::net_get;
use flexmesa_net::types::{NetGeneralInfo, NUM_RVS};
use flexmesa_net::NetWorkState;
use flexmesa_rates::ScreeningMode;

const ZONE_LND_BITS: u64 = 0x400D_5675_A4BD_6BA8;
const ZONE_LNT_BITS: u64 = 0x4030_EFA8_2F96_06DB;

const EXPECTED_EPS_NUC_BITS: u64 = 0x4087_AB06_543D_5876;
const EXPECTED_D_EPSNUC_DLND_BITS: u64 = 0x4085_615F_90DE_1C58;

const EXPECTED_PEOS_BITS: u64 = 0x437A_1E58_C8EC_54A0;
const EXPECTED_GRADA_BITS: u64 = 0x3FD9_101F_75A2_33EC;
const EXPECTED_CHI_RHO_BITS: u64 = 0x3FEF_DFB4_B2F9_9FD2;
const EXPECTED_CHI_T_BITS: u64 = 0x3FF0_4447_33CE_5C3E;
const EXPECTED_CP_BITS: u64 = 0x41B4_9A06_17B7_02B2;

const ZONE_SPECIES_HEX: [(&str, u64); 25] = [
    ("h1", 0x3FE5_9A41_C0A9_8CD5),
    ("h2", 0x3C53_CEBE_A949_2281),
    ("he3", 0x3EB9_3B37_1C95_099A),
    ("he4", 0x3FD3_7B50_D713_A3F2),
    ("li7", 0x3CBF_7A4A_B6DD_EC80),
    ("be7", 0x3DF3_1FF2_4D67_11D5),
    ("b8", 0x3C5A_1318_28AF_B6DA),
    ("c12", 0x3F05_C830_9FCE_6300),
    ("c13", 0x3EE8_106C_1184_CE78),
    ("n13", 0x3D56_9777_B9C9_7C28),
    ("n14", 0x3F77_4A45_DE01_CDB1),
    ("n15", 0x3E8C_D11A_D3E0_A5F5),
    ("o14", 0x3B09_151F_74F5_1EF0),
    ("o15", 0x3D35_463D_92CA_038C),
    ("o16", 0x3F7E_FC87_F9D6_A713),
    ("o17", 0x3F39_9533_C838_14D9),
    ("o18", 0x3E71_9DD5_6F9F_2A3F),
    ("f17", 0x3CD0_2687_3C42_81FD),
    ("f18", 0x3CEC_561C_65CC_ABFE),
    ("f19", 0x3E8A_40DC_7274_3D12),
    ("ne18", 0x3966_40D6_F1C8_1FAB),
    ("ne19", 0x3A60_851F_0459_4B74),
    ("ne20", 0x3F5E_9593_859E_4F56),
    ("mg22", 0x3E92_9A16_738A_7037),
    ("mg24", 0x3F74_4B6D_688E_1AE8),
];

struct EosWeights {
    z_int: Vec<usize>,
    azion: Vec<f64>,
    acmi: Vec<f64>,
    a_baryon: Vec<f64>,
}

struct RxnContrib<'a> {
    idx: usize,
    name: &'a str,
    d_eps_dlnd: f64,
    d_eps_drho: f64,
    eps_nuc_rate: f64,
    rate_raw: f64,
    rate_raw_drho: f64,
    rate_screened: f64,
    rate_screened_drho: f64,
    dqneu_dlnrho: f64,
}

fn ordered_f64_key(val: f64) -> u64 {
    let norm = if val == 0.0 { 0.0 } else { val };
    let bits = norm.to_bits();
    if bits >> 63 != 0 {
        !bits
    } else {
        bits ^ (1_u64 << 63)
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    ordered_f64_key(a).abs_diff(ordered_f64_key(b))
}

fn signed_bits_delta(got: f64, expected: f64) -> String {
    let gb = got.to_bits();
    let eb = expected.to_bits();
    let ulp = ulp_distance(got, expected);
    let sign = if ordered_f64_key(got) >= ordered_f64_key(expected) {
        "+"
    } else {
        "-"
    };
    format!(
        "got={got:.17e} (0x{gb:016X}), expected={expected:.17e} \
         (0x{eb:016X}), ulp={sign}{ulp}"
    )
}

fn load_hb2m_zone_xa(g: &NetGeneralInfo, catalog: &NuclideCatalog) -> Vec<f64> {
    assert_eq!(g.num_isos, ZONE_SPECIES_HEX.len());
    let mut xa = Vec::with_capacity(g.num_isos);
    for (slot, (&cid, &(expected_name, bits))) in
        g.chem_id.iter().zip(ZONE_SPECIES_HEX.iter()).enumerate()
    {
        let nuc = catalog.get_by_id(cid).expect("known nuclide");
        assert_eq!(nuc.name, expected_name, "species mismatch at slot {slot}");
        xa.push(f64::from_bits(bits));
    }
    xa
}

fn build_eos_weights(
    catalog: &NuclideCatalog,
    chem_id: &[usize],
) -> EosWeights {
    let n = chem_id.len();
    let mut z_int = Vec::with_capacity(n);
    let mut azion = Vec::with_capacity(n);
    let mut acmi = Vec::with_capacity(n);
    let mut a_baryon = Vec::with_capacity(n);
    for &cid in chem_id {
        let nuc = catalog.get_by_id(cid).expect("valid chem_id");
        let z_u32 = u32::try_from(nuc.z).expect("atomic number fits u32");
        let a_u32 = u32::try_from(nuc.z_plus_n).expect("mass number fits u32");
        z_int.push(nuc.z);
        azion.push(f64::from(z_u32));
        acmi.push(nuc.w);
        a_baryon.push(f64::from(a_u32));
    }
    EosWeights {
        z_int,
        azion,
        acmi,
        a_baryon,
    }
}

fn eval_hb2m_eos(
    g: &NetGeneralInfo,
    catalog: &NuclideCatalog,
    xa: &[f64],
    rho: f64,
    log_rho: f64,
    temp: f64,
    log_t: f64,
) -> (f64, f64, f64, f64, EosEvalOutput) {
    let comp = get_composition_info(catalog, &g.chem_id, xa, true)
        .expect("composition info");
    let weights = build_eos_weights(catalog, &g.chem_id);
    let ht = HelmTable::builtin().expect("builtin HELM table");
    let cfg = EosConfig::default();
    let eos_out = eos_dt_get_from_weights(
        ht,
        rho,
        log_rho,
        temp,
        log_t,
        comp.abar,
        comp.zbar,
        xa,
        &weights.z_int,
        &weights.azion,
        &weights.acmi,
        &weights.a_baryon,
        &cfg,
    )
    .expect("eos_dt_get_from_weights");
    (comp.abar, comp.zbar, comp.z2bar, comp.ye, eos_out)
}

fn format_eos_diagnostics(temp: f64, eos_out: &EosEvalOutput) -> String {
    let pgas = exp(eos_out.res[I_LN_PGAS]);
    let prad = ((((CRAD * temp) * temp) * temp) * temp) / 3.0;
    let peos = prad + pgas;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "EOS check:\n  Peos:   {}\n  grada:  {}\n  chiRho: {}\n  chiT:   \
         {}\n  Cp:     {}\n  eta=0x{:016X} d_eta_dlnt=0x{:016X} \
         d_eta_dlnd=0x{:016X}",
        signed_bits_delta(peos, f64::from_bits(EXPECTED_PEOS_BITS)),
        signed_bits_delta(
            eos_out.res[I_GRAD_AD],
            f64::from_bits(EXPECTED_GRADA_BITS)
        ),
        signed_bits_delta(
            eos_out.res[I_CHI_RHO],
            f64::from_bits(EXPECTED_CHI_RHO_BITS)
        ),
        signed_bits_delta(
            eos_out.res[I_CHI_T],
            f64::from_bits(EXPECTED_CHI_T_BITS)
        ),
        signed_bits_delta(eos_out.res[I_CP], f64::from_bits(EXPECTED_CP_BITS)),
        eos_out.res[I_ETA].to_bits(),
        eos_out.d_dln_t[I_ETA].to_bits(),
        eos_out.d_dlnd[I_ETA].to_bits(),
    );
    out
}

fn format_top_contributors(
    g: &NetGeneralInfo,
    w: &NetWorkState,
    rho: f64,
) -> String {
    let mut contribs: Vec<RxnContrib<'_>> = Vec::with_capacity(g.num_reactions);
    let mut sum_quad = Quad::ZERO;

    for idx in 0..g.num_reactions {
        let rate_sc = w.rate_screened[idx];
        let rate_sc_drho = w.rate_screened_drho[idx];
        let (drate_erg, dqneu_erg, dqneu_dlnrho) = if rate_sc == 0.0 {
            (0.0, 0.0, 0.0)
        } else {
            let r_val = w.screened_rate[idx] / (rate_sc * AVO);
            let drate = w.eps_nuc_rate[idx] * (rate_sc_drho / rate_sc);
            let wk1 = g.weak_reaction_index[idx];
            if wk1 > 0 {
                let wk = wk1 - 1;
                if g.weaklib_ids[wk] > 0 {
                    let dq_dlnrho = w.wk_dqneu_dln_rho[wk];
                    let dq_erg = -r_val * rate_sc * (dq_dlnrho / rho) * QCONV;
                    (drate, dq_erg, dq_dlnrho)
                } else {
                    (drate, 0.0, 0.0)
                }
            } else {
                (drate, 0.0, 0.0)
            }
        };
        let d_eps_drho = drate_erg + dqneu_erg;
        let d_eps_dlnd = d_eps_drho * rho;
        sum_quad += Quad::from_f64(d_eps_drho);
        contribs.push(RxnContrib {
            idx,
            name: g.reactions[idx].name.as_str(),
            d_eps_dlnd,
            d_eps_drho,
            eps_nuc_rate: w.eps_nuc_rate[idx],
            rate_raw: w.rate_raw[idx],
            rate_raw_drho: w.rate_raw_drho[idx],
            rate_screened: rate_sc,
            rate_screened_drho: rate_sc_drho,
            dqneu_dlnrho,
        });
    }

    contribs.sort_by(|a, b| b.d_eps_dlnd.abs().total_cmp(&a.d_eps_dlnd.abs()));
    let recon_dlnd = sum_quad.to_f64() * rho;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "Diagnostic reaction sum d_epsnuc_dlnd = {recon_dlnd:.17e} \
         (0x{:016X})",
        recon_dlnd.to_bits()
    );
    let _ = writeln!(out, "Top d_epsnuc_dlnd contributors (sorted by |val|):");
    for item in contribs.iter().take(20) {
        let _ = writeln!(
            out,
            "  [{:2}] {:<28} dlnd={:+.12e} (0x{:016X}) drho={:+.9e} \
             eps={:+.9e} r_raw={:.6e} dr_raw={:+.6e} r_sc={:.6e} \
             dr_sc={:+.6e} dqneu={:+.6e}",
            item.idx + 1,
            item.name,
            item.d_eps_dlnd,
            item.d_eps_dlnd.to_bits(),
            item.d_eps_drho,
            item.eps_nuc_rate,
            item.rate_raw,
            item.rate_raw_drho,
            item.rate_screened,
            item.rate_screened_drho,
            item.dqneu_dlnrho,
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn eval_net_variants(
    g: &NetGeneralInfo,
    xa: &[f64],
    temp: f64,
    log_t: f64,
    rho: f64,
    log_rho: f64,
    abar: f64,
    mean_z: f64,
    z_sq_bar: f64,
    ye: f64,
    eta: f64,
    d_eta_dlnt: f64,
    d_eta_dlnrho: f64,
) -> (f64, f64, NetWorkState) {
    let rate_factors = vec![1.0_f64; g.num_reactions];
    let mut dydt = vec![[Quad::ZERO; NUM_RVS]; g.num_isos];
    let mut w = NetWorkState::new(g);
    let (eps_nuc, _d_eps_dt, d_eps_drho, _eps_neu) = eval_net_with_work(
        g,
        &mut w,
        &mut dydt,
        false,
        xa,
        temp,
        log_t,
        rho,
        log_rho,
        abar,
        mean_z,
        z_sq_bar,
        ye,
        eta,
        d_eta_dlnt,
        d_eta_dlnrho,
        &rate_factors,
        1.0,
        ScreeningMode::Chugunov,
        g.use_3a_fl87,
        g.use_suzuki_weak_rates,
        false,
    )
    .expect("eval_net_with_work");

    let mut tl_dydt = vec![[Quad::ZERO; NUM_RVS]; g.num_isos];
    let (tl_eps, _tl_dt, tl_drho, _tl_neu) =
        with_thread_local_work(g, |tl_w| {
            eval_net_with_work(
                g,
                tl_w,
                &mut tl_dydt,
                false,
                xa,
                temp,
                log_t,
                rho,
                log_rho,
                abar,
                mean_z,
                z_sq_bar,
                ye,
                eta,
                d_eta_dlnt,
                d_eta_dlnrho,
                &rate_factors,
                1.0,
                ScreeningMode::Chugunov,
                g.use_3a_fl87,
                g.use_suzuki_weak_rates,
                false,
            )
        })
        .expect("with_thread_local_work");
    assert_eq!(eps_nuc.to_bits(), tl_eps.to_bits());
    assert_eq!(d_eps_drho.to_bits(), tl_drho.to_bits());

    let net_res = net_get(
        g,
        false,
        xa,
        temp,
        log_t,
        rho,
        log_rho,
        abar,
        mean_z,
        z_sq_bar,
        ye,
        eta,
        d_eta_dlnt,
        d_eta_dlnrho,
        &rate_factors,
        1.0,
        ScreeningMode::Chugunov,
    )
    .expect("net_get");
    assert_eq!(eps_nuc.to_bits(), net_res.eps_nuc.to_bits());
    assert_eq!(d_eps_drho.to_bits(), net_res.d_eps_nuc_drho.to_bits());

    (eps_nuc, d_eps_drho, w)
}

#[test]
fn test_hb2m_model64_zone1810_pp_and_cno_extras_dlnd_hex() {
    math_init();
    with_numerics_mode(NumericsMode::Strict, || {
        let g = NetGeneralInfo::from_name("pp_and_cno_extras.net")
            .expect("load pp_and_cno_extras.net");
        let catalog = NuclideCatalog::builtin_ref();
        let xa = load_hb2m_zone_xa(&g, catalog);

        let lnd = f64::from_bits(ZONE_LND_BITS);
        let lnt = f64::from_bits(ZONE_LNT_BITS);
        let rho = exp(lnd);
        let temp = exp(lnt);
        let log_rho = lnd / LN10;
        let log_t = lnt / LN10;

        let (abar, mean_z, z_sq_bar, ye, eos_out) =
            eval_hb2m_eos(&g, catalog, &xa, rho, log_rho, temp, log_t);
        let (eps_nuc, d_eps_drho, w) = eval_net_variants(
            &g,
            &xa,
            temp,
            log_t,
            rho,
            log_rho,
            abar,
            mean_z,
            z_sq_bar,
            ye,
            eos_out.res[I_ETA],
            eos_out.d_dln_t[I_ETA],
            eos_out.d_dlnd[I_ETA],
        );

        let expected_eps = f64::from_bits(EXPECTED_EPS_NUC_BITS);
        let d_eps_dlnd = d_eps_drho * rho;
        let expected_dlnd = f64::from_bits(EXPECTED_D_EPSNUC_DLND_BITS);

        if eps_nuc.to_bits() != EXPECTED_EPS_NUC_BITS
            || d_eps_dlnd.to_bits() != EXPECTED_D_EPSNUC_DLND_BITS
        {
            let eos_diag = format_eos_diagnostics(temp, &eos_out);
            let rxn_diag = format_top_contributors(&g, &w, rho);
            panic!(
                "hb_2M model 64 zone 1810 mismatch:\n  \
                 rho={rho:.17e} (0x{:016X}), T={temp:.17e} (0x{:016X})\n  \
                 eps_nuc:       {}\n  d_eps_nuc_drho={d_eps_drho:.17e} \
                 (0x{:016X})\n  d_epsnuc_dlnd: {}\n{eos_diag}{rxn_diag}",
                rho.to_bits(),
                temp.to_bits(),
                signed_bits_delta(eps_nuc, expected_eps),
                d_eps_drho.to_bits(),
                signed_bits_delta(d_eps_dlnd, expected_dlnd),
            );
        }
    });
}
