//! Conversion of raw `helmeos2` output to the 26-channel MESA `EosEvalOutput`
//! (`do_convert_helm_results`, `eval_helm_eos`, and `eval_helm_eos_zones`).
//!
//! This module ports `eos/private/eos_helm_eval.f90` (`get_helm_for_eosdt` and
//! `do_convert_helm_results`).

use flexmesa_const::AVO;
use rayon::prelude::*;

use crate::helm::eval::{HelmConfig, helmeos2};
use crate::helm::table::HelmTable;
use crate::helm::{
    H_CHID, H_CHIT, H_CP, H_CV, H_DCHIDDA, H_DCHIDDD, H_DCHIDDT, H_DCHIDDZ,
    H_DCHITDA, H_DCHITDD, H_DCHITDT, H_DCHITDZ, H_DCPDA, H_DCPDD, H_DCPDT,
    H_DCPDZ, H_DCVDA, H_DCVDD, H_DCVDT, H_DCVDZ, H_DEA, H_DED, H_DEDA,
    H_DEDD, H_DEDT, H_DEDZ, H_DET, H_DETAA, H_DETAD, H_DETAT, H_DETAZ, H_DEZ,
    H_DGAM1DA, H_DGAM1DD, H_DGAM1DT, H_DGAM1DZ, H_DGAM3DA, H_DGAM3DD,
    H_DGAM3DT, H_DGAM3DZ, H_DNABDA, H_DNABDD, H_DNABDT, H_DNABDZ, H_DPGASA,
    H_DPGASD, H_DPGAST, H_DPGASZ, H_DSA, H_DSD, H_DSDA, H_DSDD, H_DSDT,
    H_DSDZ, H_DST, H_DSTA, H_DSTT, H_DSTZ, H_DSZ, H_DXNEA, H_DXNED, H_DXNET,
    H_DXNEZ, H_ETAELE, H_ETOT, H_GAM1, H_GAM3, H_NABAD, H_PGAS, H_STOT,
    H_XNE, NUM_HELM_RESULTS,
};
use crate::types::{
    EosEvalOutput, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_DE_DRHO, I_DS_DRHO,
    I_DS_DT, I_ETA, I_FRAC_HELM, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LN_E,
    I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, NUM_EOS_BASIC_RESULTS,
};

/// Minimum zone count for parallel `rayon` dispatch in `eval_helm_eos_zones`.
const PARALLEL_ZONE_THRESHOLD: usize = 64;

/// Converts raw `helmeos2` output into the 26-channel MESA `EosEvalOutput` and
/// composition partial derivatives `(d_dabar, d_dzbar)`
/// (`do_convert_helm_results` in `eos/private/eos_helm_eval.f90`).
#[must_use]
pub fn do_convert_helm_results(
    abar: f64,
    zbar: f64,
    rho: f64,
    t: f64,
    helm_res: &[f64; NUM_HELM_RESULTS],
) -> (
    EosEvalOutput,
    [f64; NUM_EOS_BASIC_RESULTS],
    [f64; NUM_EOS_BASIC_RESULTS],
) {
    let mut out = EosEvalOutput::zeros(0);
    let mut d_dabar = [0.0; NUM_EOS_BASIC_RESULTS];
    let mut d_dzbar = [0.0; NUM_EOS_BASIC_RESULTS];

    let energy = helm_res[H_ETOT - 1];
    let entropy = helm_res[H_STOT - 1];
    let pgas = helm_res[H_PGAS - 1];
    let mu = abar / (1.0 + zbar);
    let free_e = helm_res[H_XNE - 1].max(1.0e-99) / (AVO * rho);

    out.res[I_LN_PGAS] = pgas.ln();
    out.res[I_LN_E] = energy.ln();
    out.res[I_LN_S] = entropy.ln();
    out.res[I_MU] = mu;
    out.res[I_LN_FREE_E] = free_e.ln();
    out.res[I_ETA] = helm_res[H_ETAELE - 1];
    out.res[I_GRAD_AD] = helm_res[H_NABAD - 1];
    out.res[I_CHI_RHO] = helm_res[H_CHID - 1];
    out.res[I_CHI_T] = helm_res[H_CHIT - 1];
    out.res[I_CP] = helm_res[H_CP - 1];
    out.res[I_CV] = helm_res[H_CV - 1];
    out.res[I_DE_DRHO] = helm_res[H_DED - 1];
    out.res[I_DS_DT] = helm_res[H_DST - 1];
    out.res[I_DS_DRHO] = helm_res[H_DSD - 1];
    out.res[I_GAMMA1] = helm_res[H_GAM1 - 1];
    out.res[I_GAMMA3] = helm_res[H_GAM3 - 1];
    out.res[I_FRAC_HELM] = 1.0;

    out.d_dlnd[I_LN_PGAS] = helm_res[H_DPGASD - 1] * rho / pgas;
    out.d_dlnd[I_LN_E] = helm_res[H_DED - 1] * rho / energy;
    out.d_dlnd[I_LN_S] = helm_res[H_DSD - 1] * rho / entropy;
    out.d_dlnd[I_LN_FREE_E] = helm_res[H_DXNED - 1] / (AVO * free_e) - 1.0;
    out.d_dlnd[I_ETA] = helm_res[H_DETAD - 1] * rho;
    out.d_dlnd[I_GRAD_AD] = helm_res[H_DNABDD - 1] * rho;
    out.d_dlnd[I_CHI_RHO] = helm_res[H_DCHIDDD - 1] * rho;
    out.d_dlnd[I_CHI_T] = helm_res[H_DCHITDD - 1] * rho;
    out.d_dlnd[I_CP] = helm_res[H_DCPDD - 1] * rho;
    out.d_dlnd[I_CV] = helm_res[H_DCVDD - 1] * rho;
    out.d_dlnd[I_DE_DRHO] = helm_res[H_DEDD - 1] * rho;
    out.d_dlnd[I_DS_DT] = helm_res[H_DSDT - 1] * rho;
    out.d_dlnd[I_DS_DRHO] = helm_res[H_DSDD - 1] * rho;
    out.d_dlnd[I_GAMMA1] = helm_res[H_DGAM1DD - 1] * rho;
    out.d_dlnd[I_GAMMA3] = helm_res[H_DGAM3DD - 1] * rho;

    out.d_dln_t[I_LN_PGAS] = helm_res[H_DPGAST - 1] * t / pgas;
    out.d_dln_t[I_LN_E] = helm_res[H_DET - 1] * t / energy;
    out.d_dln_t[I_LN_S] = helm_res[H_DST - 1] * t / entropy;
    out.d_dln_t[I_LN_FREE_E] =
        (helm_res[H_DXNET - 1] * t / (AVO * rho)) / free_e;
    out.d_dln_t[I_ETA] = helm_res[H_DETAT - 1] * t;
    out.d_dln_t[I_GRAD_AD] = helm_res[H_DNABDT - 1] * t;
    out.d_dln_t[I_CHI_RHO] = helm_res[H_DCHIDDT - 1] * t;
    out.d_dln_t[I_CHI_T] = helm_res[H_DCHITDT - 1] * t;
    out.d_dln_t[I_CP] = helm_res[H_DCPDT - 1] * t;
    out.d_dln_t[I_CV] = helm_res[H_DCVDT - 1] * t;
    out.d_dln_t[I_DE_DRHO] = helm_res[H_DEDT - 1] * t;
    out.d_dln_t[I_DS_DT] = helm_res[H_DSTT - 1] * t;
    out.d_dln_t[I_DS_DRHO] = helm_res[H_DSDT - 1] * t;
    out.d_dln_t[I_GAMMA1] = helm_res[H_DGAM1DT - 1] * t;
    out.d_dln_t[I_GAMMA3] = helm_res[H_DGAM3DT - 1] * t;

    d_dabar[I_LN_PGAS] = helm_res[H_DPGASA - 1] / pgas;
    d_dabar[I_LN_E] = helm_res[H_DEA - 1] / energy;
    d_dabar[I_LN_S] = helm_res[H_DSA - 1] / entropy;
    d_dabar[I_LN_FREE_E] = (helm_res[H_DXNEA - 1] / (AVO * rho)) / free_e;
    d_dabar[I_ETA] = helm_res[H_DETAA - 1];
    d_dabar[I_GRAD_AD] = helm_res[H_DNABDA - 1];
    d_dabar[I_CHI_RHO] = helm_res[H_DCHIDDA - 1];
    d_dabar[I_CHI_T] = helm_res[H_DCHITDA - 1];
    d_dabar[I_CP] = helm_res[H_DCPDA - 1];
    d_dabar[I_CV] = helm_res[H_DCVDA - 1];
    d_dabar[I_DE_DRHO] = helm_res[H_DEDA - 1];
    d_dabar[I_DS_DT] = helm_res[H_DSTA - 1];
    d_dabar[I_DS_DRHO] = helm_res[H_DSDA - 1];
    d_dabar[I_GAMMA1] = helm_res[H_DGAM1DA - 1];
    d_dabar[I_GAMMA3] = helm_res[H_DGAM3DA - 1];

    d_dzbar[I_LN_PGAS] = helm_res[H_DPGASZ - 1] / pgas;
    d_dzbar[I_LN_E] = helm_res[H_DEZ - 1] / energy;
    d_dzbar[I_LN_S] = helm_res[H_DSZ - 1] / entropy;
    d_dzbar[I_LN_FREE_E] = (helm_res[H_DXNEZ - 1] / (AVO * rho)) / free_e;
    d_dzbar[I_ETA] = helm_res[H_DETAZ - 1];
    d_dzbar[I_GRAD_AD] = helm_res[H_DNABDZ - 1];
    d_dzbar[I_CHI_RHO] = helm_res[H_DCHIDDZ - 1];
    d_dzbar[I_CHI_T] = helm_res[H_DCHITDZ - 1];
    d_dzbar[I_CP] = helm_res[H_DCPDZ - 1];
    d_dzbar[I_CV] = helm_res[H_DCVDZ - 1];
    d_dzbar[I_DE_DRHO] = helm_res[H_DEDZ - 1];
    d_dzbar[I_DS_DT] = helm_res[H_DSTZ - 1];
    d_dzbar[I_DS_DRHO] = helm_res[H_DSDZ - 1];
    d_dzbar[I_GAMMA1] = helm_res[H_DGAM1DZ - 1];
    d_dzbar[I_GAMMA3] = helm_res[H_DGAM3DZ - 1];

    (out, d_dabar, d_dzbar)
}

/// Evaluates the 26-channel `HELM` equation of state (`get_helm_for_eosdt` in
/// `eos/private/eos_helm_eval.f90`) using the built-in [`HelmTable`].
///
/// # Errors
/// Returns `Err(-1)` if `helmeos2` encounters non-positive thermodynamic state
/// variables or invalid susceptibilities.
pub fn eval_helm_eos(
    temp: f64,
    den: f64,
    abar: f64,
    zbar: f64,
) -> Result<
    (
        EosEvalOutput,
        [f64; NUM_EOS_BASIC_RESULTS],
        [f64; NUM_EOS_BASIC_RESULTS],
    ),
    i32,
> {
    let table = HelmTable::builtin();
    let config = HelmConfig::default();
    let (helm_res, _off_table) = helmeos2(
        table,
        temp,
        temp.log10(),
        den,
        den.log10(),
        abar,
        zbar,
        &config,
    )?;
    Ok(do_convert_helm_results(abar, zbar, den, temp, &helm_res))
}

/// Evaluates `eval_helm_eos` across a batch of stellar zones in parallel when
/// `n_zones` exceeds the parallel threshold (`PARALLEL_ZONE_THRESHOLD = 64`).
#[must_use]
pub fn eval_helm_eos_zones(
    temps: &[f64],
    dens: &[f64],
    abars: &[f64],
    zbars: &[f64],
) -> Vec<EosEvalOutput> {
    let n = temps.len().min(dens.len()).min(abars.len()).min(zbars.len());
    let eval_one = |i: usize| {
        eval_helm_eos(temps[i], dens[i], abars[i], zbars[i]).map_or_else(
            |_| EosEvalOutput {
                res: [f64::NAN; NUM_EOS_BASIC_RESULTS],
                d_dlnd: [f64::NAN; NUM_EOS_BASIC_RESULTS],
                d_dln_t: [f64::NAN; NUM_EOS_BASIC_RESULTS],
                d_dxa: Vec::new(),
            },
            |(out, _, _)| out,
        )
    };
    if n >= PARALLEL_ZONE_THRESHOLD {
        (0..n).into_par_iter().map(eval_one).collect()
    } else {
        (0..n).map(eval_one).collect()
    }
}
