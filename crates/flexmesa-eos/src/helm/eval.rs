//! Full `HELM` equation-of-state evaluation (`helmeos2` and `HelmConfig`).
//!
//! This module ports `eos/private/helm.f90`, `eos/private/helm_sum_totals.dek`,
//! `eos/private/helm_gammas.dek`, and `eos/private/helm_store_results.dek`.

use flexmesa_const::CLIGHT;

use crate::helm::components::{HelmThermoVar, eval_ideal_ions, eval_radiation};
use crate::helm::coulomb::eval_helm_coulomb;
use crate::helm::elec_pos::eval_electron_positron;
use crate::helm::table::{
    HELM_LOGDHI, HELM_LOGDLO, HELM_LOGTHI, HELM_LOGTLO, HelmTable,
};
use crate::helm::{
    H_CHID, H_CHIT, H_CP, H_CS, H_CV, H_DCHIDDA, H_DCHIDDD, H_DCHIDDT,
    H_DCHIDDZ, H_DCHITDA, H_DCHITDD, H_DCHITDT, H_DCHITDZ, H_DCPDA, H_DCPDD,
    H_DCPDT, H_DCPDZ, H_DCVDA, H_DCVDD, H_DCVDT, H_DCVDZ, H_DEA, H_DED, H_DEDA,
    H_DEDD, H_DEDT, H_DEDZ, H_DET, H_DETAA, H_DETAD, H_DETAT, H_DETAZ, H_DEZ,
    H_DGAM1DA, H_DGAM1DD, H_DGAM1DT, H_DGAM1DZ, H_DGAM2DA, H_DGAM2DD,
    H_DGAM2DT, H_DGAM2DZ, H_DGAM3DA, H_DGAM3DD, H_DGAM3DT, H_DGAM3DZ,
    H_DNABDA, H_DNABDD, H_DNABDT, H_DNABDZ, H_DPA, H_DPD, H_DPGASA, H_DPGASD,
    H_DPGASDD, H_DPGAST, H_DPGASDT, H_DPGASZ, H_DPRADD, H_DPRADT, H_DPT,
    H_DPZ, H_DSA, H_DSD, H_DSDA, H_DSDD, H_DSDT, H_DSDZ, H_DST, H_DSTA,
    H_DSTT, H_DSTZ, H_DSZ, H_DXNEA, H_DXNED, H_DXNET, H_DXNEZ, H_EGAS, H_ERAD,
    H_ETAELE, H_ETOT, H_GAM1, H_GAM2, H_GAM3, H_NABAD, H_PGAS, H_PLASG,
    H_PRAD, H_PTOT, H_SGAS, H_SRAD, H_STOT, H_VALID, H_XNE, H_XNI,
    NUM_HELM_RESULTS,
};

/// Configuration parameters for the `HELM` equation of state (`eos.defaults`).
#[derive(Debug, Clone, Copy)]
pub struct HelmConfig {
    /// Temperature threshold ($\mathrm{K}$) below which Coulomb terms roll off.
    pub coulomb_temp_cut: f64,
    /// Density threshold ($\mathrm{g\,cm^{-3}}$) below which Coulomb terms
    /// roll off.
    pub coulomb_den_cut: f64,
    /// Include blackbody radiation pressure, energy, and entropy.
    pub include_radiation: bool,
    /// Include electron-positron biquintic/bicubic table contributions.
    pub include_elec_pos: bool,
    /// Clip out-of-bounds $(\rho Y_e, T)$ queries to `HELM` table boundaries.
    pub clip_to_table_boundaries: bool,
}

impl Default for HelmConfig {
    fn default() -> Self {
        Self {
            coulomb_temp_cut: -1.0e99,
            coulomb_den_cut: -1.0e99,
            include_radiation: true,
            include_elec_pos: true,
            clip_to_table_boundaries: true,
        }
    }
}

/// Evaluates the raw 409-element `HELM` result vector (`helmeos2` in
/// `eos/private/helm.f90`).
///
/// # Arguments
/// * `table` - Reference to the [`HelmTable`].
/// * `temp_in` - Temperature $T$ ($\mathrm{K}$).
/// * `log_temp_in` - $\log_{10}(T/\mathrm{K})$.
/// * `den_in` - Mass density $\rho$ ($\mathrm{g\,cm^{-3}}$).
/// * `log_den_in` - $\log_{10}(\rho/\mathrm{g\,cm^{-3}})$.
/// * `abar` - Mean atomic weight $\bar{A}$ ($\mathrm{g\,mol^{-1}}$).
/// * `zbar` - Mean ionic charge $\bar{Z}$ ($e$).
/// * `config` - [`HelmConfig`] options.
///
/// # Returns
/// `(helm_res, off_table)` on success.
///
/// # Errors
/// Returns `Err(ierr)` if out of bounds or thermodynamically inconsistent
/// (`dsepdt < 0`, `cv < 0`, `gam1 < 0`).
#[allow(clippy::too_many_arguments)]
pub fn helmeos2(
    table: &HelmTable,
    temp_in: f64,
    log_temp_in: f64,
    den_in: f64,
    log_den_in: f64,
    abar: f64,
    zbar: f64,
    config: &HelmConfig,
) -> Result<([f64; NUM_HELM_RESULTS], bool), i32> {
    if !(abar > 0.0 && zbar > 0.0 && temp_in > 0.0 && den_in > 0.0) {
        return Err(1);
    }
    let ytot1 = 1.0 / abar;
    let ye = ytot1 * zbar;

    let mut temp = temp_in;
    let mut log_temp = log_temp_in;
    let mut den = den_in;
    let mut din = ye * den;
    let mut off_table = false;

    if config.include_elec_pos {
        let templo = 1.0e3;
        let temphi = 1.0e13;
        let denlo = 1.0e-12;
        let denhi = 1.0e15;
        if config.clip_to_table_boundaries {
            if log_temp <= HELM_LOGTLO {
                log_temp = HELM_LOGTLO;
                temp = templo;
                off_table = true;
            } else if log_temp >= HELM_LOGTHI {
                log_temp = HELM_LOGTHI;
                temp = temphi;
                off_table = true;
            }
            let log_din = log_den_in + ye.log10();
            if log_din <= HELM_LOGDLO {
                din = denlo;
                den = din / ye;
                off_table = true;
            } else if log_din >= HELM_LOGDHI {
                din = denhi;
                den = din / ye;
                off_table = true;
            }
        }
        if temp < templo || din < denlo || temp > temphi || din > denhi {
            return Err(1);
        }
    }

    let (prad, erad, srad) =
        eval_radiation(temp, den, config.include_radiation);
    let (pion, eion, sion, xni, _etaion) = eval_ideal_ions(temp, den, abar);
    let (pele, eele, sele, etaele, xnefer) = eval_electron_positron(
        table,
        temp,
        log_temp,
        den,
        abar,
        zbar,
        config.include_elec_pos,
    )?;

    let (pcoul, ecoul, scoul, plasg) = if config.include_elec_pos {
        eval_helm_coulomb(
            temp,
            den,
            abar,
            zbar,
            &xni,
            &pion,
            config.coulomb_temp_cut,
            config.coulomb_den_cut,
        )?
    } else {
        (
            HelmThermoVar::default(),
            HelmThermoVar::default(),
            HelmThermoVar::default(),
            0.0,
        )
    };

    let pgas = pion.add(&pele).add(&pcoul);
    let egas = eion.add(&eele).add(&ecoul);
    let sgas = sion.add(&sele).add(&scoul);
    if pgas.val < 1.0e-20 || egas.val < 1.0e-20 || sgas.val < 1.0e-20 {
        return Err(1);
    }

    let pres = prad.add(&pgas);
    let ener = erad.add(&egas);
    let entr = srad.add(&sgas);

    let mut res = [0.0_f64; NUM_HELM_RESULTS];
    store_gammas_and_results(
        temp, den, &pres, &ener, &entr, &pgas, &egas, &sgas, &prad, &erad,
        &srad, &etaele, &xni, &xnefer, plasg, &mut res,
    )?;
    res[H_VALID - 1] = 1.0;
    Ok((res, off_table))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn store_gammas_and_results(
    temp: f64,
    den: f64,
    pres: &HelmThermoVar,
    ener: &HelmThermoVar,
    entr: &HelmThermoVar,
    pgas: &HelmThermoVar,
    egas: &HelmThermoVar,
    sgas: &HelmThermoVar,
    prad: &HelmThermoVar,
    erad: &HelmThermoVar,
    srad: &HelmThermoVar,
    etaele: &HelmThermoVar,
    xni: &HelmThermoVar,
    xnefer: &HelmThermoVar,
    plasg: f64,
    res: &mut [f64; NUM_HELM_RESULTS],
) -> Result<(), i32> {
    let tempinv = 1.0 / temp;
    let deninv = 1.0 / den;
    let presinv = 1.0 / pres.val;

    let chit = temp * presinv * pres.dt;
    let dchitdd =
        temp * presinv * (pres.ddt - pres.dt * presinv * pres.dd);
    let dchitdt =
        chit * (tempinv - presinv * pres.dt) + temp * presinv * pres.dtt;
    let dchitda =
        temp * presinv * (pres.dta - pres.dt * presinv * pres.da);
    let dchitdz =
        temp * presinv * (pres.dtz - pres.dt * presinv * pres.dz);

    let chid = den * presinv * pres.dd;
    let dchiddd =
        chid * (deninv - presinv * pres.dd) + den * presinv * pres.ddd;
    let dchiddt =
        den * presinv * (pres.ddt - pres.dd * presinv * pres.dt);
    let dchidda =
        den * presinv * (pres.dda - pres.dd * presinv * pres.da);
    let dchiddz =
        den * presinv * (pres.ddz - pres.dd * presinv * pres.dz);

    let cv = ener.dt;
    if cv < 0.0 {
        return Err(-1);
    }
    let cvinv = 1.0 / cv;
    let dcvdd = ener.ddt;
    let dcvdt = ener.dtt;
    let dcvda = ener.dta;
    let dcvdz = ener.dtz;

    let x = pres.val * deninv * tempinv * cvinv * chit;
    let dxdd = x
        * (presinv * pres.dd - deninv - cvinv * dcvdd + dchitdd / chit);
    let dxdt = x
        * (presinv * pres.dt - tempinv - cvinv * dcvdt + dchitdt / chit);
    let dxda = x * (presinv * pres.da - cvinv * dcvda + dchitda / chit);
    let dxdz = x * (presinv * pres.dz - cvinv * dcvdz + dchitdz / chit);

    let gam3 = x + 1.0;
    let dgam3dd = dxdd;
    let dgam3dt = dxdt;
    let dgam3da = dxda;
    let dgam3dz = dxdz;

    let gam1 = chit * x + chid;
    if gam1 < 0.0 {
        return Err(-1);
    }
    let dgam1dd = dchitdd * x + chit * dxdd + dchiddd;
    let dgam1dt = dchitdt * x + chit * dxdt + dchiddt;
    let dgam1da = dchitda * x + chit * dxda + dchidda;
    let dgam1dz = dchitdz * x + chit * dxdz + dchiddz;

    let nabad = x / gam1;
    let dnabdd = (dxdd - nabad * dgam1dd) / gam1;
    let dnabdt = (dxdt - nabad * dgam1dt) / gam1;
    let dnabda = (dxda - nabad * dgam1da) / gam1;
    let dnabdz = (dxdz - nabad * dgam1dz) / gam1;

    let gam2 = 1.0 / (1.0 - nabad);
    let y = gam2 * gam2;
    let dgam2dd = dnabdd * y;
    let dgam2dt = dnabdt * y;
    let dgam2da = dnabda * y;
    let dgam2dz = dnabdz * y;

    let cp = cv * gam1 / chid;
    let z = 1.0 / chid;
    let dcpdd = cp * (cvinv * dcvdd + dgam1dd / gam1 - z * dchiddd);
    let dcpdt = cp * (cvinv * dcvdt + dgam1dt / gam1 - z * dchiddt);
    let dcpda = cp * (cvinv * dcvda + dgam1da / gam1 - z * dchidda);
    let dcpdz = cp * (cvinv * dcvdz + dgam1dz / gam1 - z * dchiddz);

    let z_cs = 1.0 + (ener.val + CLIGHT * CLIGHT) * den / pres.val;
    let cs = CLIGHT * (gam1 / z_cs).sqrt();

    res[H_PTOT - 1] = pres.val;
    res[H_DPD - 1] = pres.dd;
    res[H_DPT - 1] = pres.dt;
    res[H_DPA - 1] = pres.da;
    res[H_DPZ - 1] = pres.dz;
    res[H_ETOT - 1] = ener.val;
    res[H_DED - 1] = ener.dd;
    res[H_DET - 1] = ener.dt;
    res[H_DEA - 1] = ener.da;
    res[H_DEZ - 1] = ener.dz;
    res[H_DEDD - 1] = ener.ddd;
    res[H_DEDT - 1] = ener.ddt;
    res[H_DEDA - 1] = ener.dda;
    res[H_DEDZ - 1] = ener.ddz;
    res[H_STOT - 1] = entr.val;
    res[H_DSD - 1] = entr.dd;
    res[H_DST - 1] = entr.dt;
    res[H_DSA - 1] = entr.da;
    res[H_DSZ - 1] = entr.dz;
    res[H_DSDD - 1] = entr.ddd;
    res[H_DSDT - 1] = entr.ddt;
    res[H_DSDA - 1] = entr.dda;
    res[H_DSDZ - 1] = entr.ddz;
    res[H_DSTT - 1] = entr.dtt;
    res[H_DSTA - 1] = entr.dta;
    res[H_DSTZ - 1] = entr.dtz;

    res[H_PGAS - 1] = pgas.val;
    res[H_DPGASD - 1] = pgas.dd;
    res[H_DPGAST - 1] = pgas.dt;
    res[H_DPGASA - 1] = pgas.da;
    res[H_DPGASZ - 1] = pgas.dz;
    res[H_DPGASDD - 1] = pgas.ddd;
    res[H_DPGASDT - 1] = pgas.ddt;
    res[H_EGAS - 1] = egas.val;
    res[H_SGAS - 1] = sgas.val;

    res[H_PRAD - 1] = prad.val;
    res[H_DPRADD - 1] = prad.dd;
    res[H_DPRADT - 1] = prad.dt;
    res[H_ERAD - 1] = erad.val;
    res[H_SRAD - 1] = srad.val;

    res[H_ETAELE - 1] = etaele.val;
    res[H_DETAD - 1] = etaele.dd;
    res[H_DETAT - 1] = etaele.dt;
    res[H_DETAA - 1] = etaele.da;
    res[H_DETAZ - 1] = etaele.dz;
    res[H_XNI - 1] = xni.val;
    res[H_XNE - 1] = xnefer.val;
    res[H_DXNED - 1] = xnefer.dd;
    res[H_DXNET - 1] = xnefer.dt;
    res[H_DXNEA - 1] = xnefer.da;
    res[H_DXNEZ - 1] = xnefer.dz;
    res[H_PLASG - 1] = plasg;

    res[H_CHIT - 1] = chit;
    res[H_DCHITDD - 1] = dchitdd;
    res[H_DCHITDT - 1] = dchitdt;
    res[H_DCHITDA - 1] = dchitda;
    res[H_DCHITDZ - 1] = dchitdz;
    res[H_CHID - 1] = chid;
    res[H_DCHIDDD - 1] = dchiddd;
    res[H_DCHIDDT - 1] = dchiddt;
    res[H_DCHIDDA - 1] = dchidda;
    res[H_DCHIDDZ - 1] = dchiddz;
    res[H_CV - 1] = cv;
    res[H_DCVDD - 1] = dcvdd;
    res[H_DCVDT - 1] = dcvdt;
    res[H_DCVDA - 1] = dcvda;
    res[H_DCVDZ - 1] = dcvdz;
    res[H_CP - 1] = cp;
    res[H_DCPDD - 1] = dcpdd;
    res[H_DCPDT - 1] = dcpdt;
    res[H_DCPDA - 1] = dcpda;
    res[H_DCPDZ - 1] = dcpdz;
    res[H_GAM1 - 1] = gam1;
    res[H_DGAM1DD - 1] = dgam1dd;
    res[H_DGAM1DT - 1] = dgam1dt;
    res[H_DGAM1DA - 1] = dgam1da;
    res[H_DGAM1DZ - 1] = dgam1dz;
    res[H_GAM2 - 1] = gam2;
    res[H_DGAM2DD - 1] = dgam2dd;
    res[H_DGAM2DT - 1] = dgam2dt;
    res[H_DGAM2DA - 1] = dgam2da;
    res[H_DGAM2DZ - 1] = dgam2dz;
    res[H_GAM3 - 1] = gam3;
    res[H_DGAM3DD - 1] = dgam3dd;
    res[H_DGAM3DT - 1] = dgam3dt;
    res[H_DGAM3DA - 1] = dgam3da;
    res[H_DGAM3DZ - 1] = dgam3dz;
    res[H_NABAD - 1] = nabad;
    res[H_DNABDD - 1] = dnabdd;
    res[H_DNABDT - 1] = dnabdt;
    res[H_DNABDA - 1] = dnabda;
    res[H_DNABDZ - 1] = dnabdz;
    res[H_CS - 1] = cs;
    Ok(())
}
