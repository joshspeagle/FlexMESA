//! Multi-component fully ionized electron-ion plasma combiner (`MELANGE9`)
//! for the Potekhin-Chabrier (`PC`) equation of state.
//!
//! Reference: Potekhin & Chabrier (2010, 2013); upstream Fortran MESA
//! `eos/private/pc_eos.f90` lines 210-482.

use flexmesa_autodiff::AutoDiffReal2VarOrder1;
use flexmesa_const::PI;

use crate::pc::coulomb_ion::PcIonContrib;
use crate::pc::coulomb_screen::{cormix, eosfi8};
use crate::pc::elect::elect11;
use crate::pc::ferinv::chemfit;
use crate::types::EosError;

/// Output of `MELANGE9` (`pc_eos.f90` lines 210-482).
#[derive(Debug, Clone, Copy)]
pub struct Melange9Output {
    /// Radiative pressure divided by `n_i k_B T`.
    pub prad_nkt: AutoDiffReal2VarOrder1,
    /// Electron number density in atomic units (`a_0^{-3}`).
    pub dens: AutoDiffReal2VarOrder1,
    /// Mean ion charge `\langle Z \rangle`.
    pub zmean: f64,
    /// Mean atomic mass `\langle A \rangle` (`\mathrm{amu}`).
    pub cmi_mean: f64,
    /// Mean-square ion charge `\langle Z^2 \rangle`.
    pub z2mean: f64,
    /// Effective mean ion Coulomb coupling parameter `\Gamma_i`.
    pub gami_mean: AutoDiffReal2VarOrder1,
    /// Dimensionless electron chemical potential `\chi = \mu_e / (k_B T)`.
    pub chi: AutoDiffReal2VarOrder1,
    /// Effective ion quantum parameter `T_p / T`.
    pub tpt: AutoDiffReal2VarOrder1,
    /// Phase regime indicator (`0` liquid, `1` solid, `-1` liquid-solid blend).
    pub liqsol: i32,
    /// Plasma pressure divided by `n_i k_B T`.
    pub p_nkt: AutoDiffReal2VarOrder1,
    /// Plasma internal energy per ion divided by `k_B T`.
    pub u_nkt: AutoDiffReal2VarOrder1,
    /// Plasma entropy per ion divided by `k_B`.
    pub s_nk: AutoDiffReal2VarOrder1,
    /// Plasma isochoric heat capacity per ion divided by `k_B`.
    pub cv: AutoDiffReal2VarOrder1,
    /// Logarithmic density derivative `\chi_\rho = (\partial \ln P / \partial \ln \rho)_T`.
    pub chir: AutoDiffReal2VarOrder1,
    /// Logarithmic temperature derivative `\chi_T = (\partial \ln P / \partial \ln T)_\rho`.
    pub chit: AutoDiffReal2VarOrder1,
}

/// Evaluate `MELANGE9` (`pc_eos.f90` lines 210-482) for a multi-component
/// fully ionized electron-ion plasma mixture.
pub fn melange9(
    ay_in: &[f64],
    azion: &[f64],
    acmi: &[f64],
    rho: AutoDiffReal2VarOrder1,
    temp: AutoDiffReal2VarOrder1,
    gami_lo: f64,
    gami_hi: f64,
) -> Result<Melange9Output, EosError> {
    const TINY: f64 = 1.0e-7;
    const AUM: f64 = 1822.888;
    const RSIMELT: f64 = 140.0;
    const RAD: f64 = 2.5568570411948021e-07;

    let nmix = ay_in.len().min(azion.len()).min(acmi.len());
    let mut ay = ay_in[..nmix].to_vec();
    let y_sum: f64 = ay.iter().sum();
    if y_sum <= 0.0 {
        return Err(EosError::InvalidState {
            context: "melange9",
            detail: "non-positive ion abundance sum".to_string(),
        });
    }
    if (y_sum - 1.0).abs() > TINY {
        for val in &mut ay {
            *val /= y_sum;
        }
    }

    let mut zmean = 0.0_f64;
    let mut z2mean = 0.0_f64;
    let mut z52 = 0.0_f64;
    let mut z53 = 0.0_f64;
    let mut z73 = 0.0_f64;
    let mut z321 = 0.0_f64;
    let mut cmi_mean = 0.0_f64;

    for ix in 0..nmix {
        if ay[ix] < TINY {
            continue;
        }
        let z = azion[ix];
        let y = ay[ix];
        zmean += y * z;
        z2mean += y * z * z;
        let z13 = z.powf(1.0 / 3.0);
        z53 += y * z13.powi(5);
        z73 += y * z13.powi(7);
        z52 += y * z.sqrt().powi(5);
        z321 += y * z * (z + 1.0).sqrt().powi(3);
        cmi_mean += y * acmi[ix];
    }

    let uint_rad = temp.pow2().pow2() * RAD;
    let press_rad = uint_rad / 3.0;

    let dens0 = rho / 11.20587 * (zmean / cmi_mean);
    let chi = chemfit(dens0, temp);
    let el = elect11(temp, chi);
    let dens = el.dens;
    let dte = dens * temp;
    let presse = el.pe_id * dte;
    let uinte = el.ue_id * dte;

    let rs = (dens.recip() * (0.75 / PI)).powf(1.0 / 3.0);
    let rsi = rs * (cmi_mean * z73 * AUM);
    let game = (rs * temp).recip();
    let gami_mean = game * z53;

    let (liqsol, alfa, beta) = if rsi.val < RSIMELT || gami_mean.val < gami_lo {
        (
            0,
            AutoDiffReal2VarOrder1::zero(),
            AutoDiffReal2VarOrder1::constant(1.0),
        )
    } else if gami_mean.val > gami_hi {
        (
            1,
            AutoDiffReal2VarOrder1::constant(1.0),
            AutoDiffReal2VarOrder1::zero(),
        )
    } else {
        let a = (gami_mean - gami_lo) / (gami_hi - gami_lo);
        let b = -a + 1.0;
        (-1, a, b)
    };

    let mut uint = uinte;
    let mut press = presse;
    let mut cv_tot = el.cve * dens;
    let mut s_tot = el.se_id * dens;
    let mut pdlt = presse * el.chite;
    let mut pdlr = presse * el.chire;
    let densi = dens / zmean;
    let pressi = densi * temp;
    let mut tpt2 = AutoDiffReal2VarOrder1::zero();
    let ctp = (temp * temp).recip() * (4.0 * PI / AUM);

    for ix in 0..nmix {
        if ay[ix] < TINY {
            continue;
        }
        let zion = azion[ix];
        if zion == 0.0 {
            continue;
        }
        let cmi = acmi[ix];
        let gami = game * zion.powf(5.0 / 3.0);
        let dni = densi * ay[ix];
        let pri = dni * temp;
        let c2 = if liqsol == 0 || liqsol == 1 {
            eosfi8(liqsol, cmi, zion, rs, gami)?.1
        } else {
            let c2_0 = eosfi8(0, cmi, zion, rs, gami)?.1;
            let c2_1 = eosfi8(1, cmi, zion, rs, gami)?.1;
            PcIonContrib {
                f: alfa * c2_1.f + beta * c2_0.f,
                u: alfa * c2_1.u + beta * c2_0.u,
                p: alfa * c2_1.p + beta * c2_0.p,
                s: alfa * c2_1.s + beta * c2_0.s,
                cv: alfa * c2_1.cv + beta * c2_0.cv,
                pdt: alfa * c2_1.pdt + beta * c2_0.pdt,
                pdr: alfa * c2_1.pdr + beta * c2_0.pdr,
            }
        };
        uint = uint + c2.u * pri;
        s_tot = s_tot + dni * (c2.s - ay[ix].ln());
        press = press + c2.p * pri;
        cv_tot = cv_tot + dni * c2.cv;
        pdlt = pdlt + pri * c2.pdt;
        pdlr = pdlr + pri * c2.pdr;
        tpt2 = tpt2 + ctp * dni * (zion * zion / cmi);
    }

    let tpt = tpt2.sqrt();
    if liqsol == 0 {
        let fwk = tpt2 / 24.0;
        let uwk = fwk * 2.0;
        uint = uint + uwk * pressi;
        s_tot = s_tot + fwk * densi;
        press = press + uwk * 0.5 * pressi;
        cv_tot = cv_tot - uwk * densi;
        pdlt = pdlt - fwk * pressi;
        pdlr = pdlr + uwk * pressi;
    }

    let mix = if liqsol == 0 {
        cormix(rs, game, zmean, z2mean, z52, z53, z321)
    } else {
        let mut fmix = AutoDiffReal2VarOrder1::zero();
        for i in 0..nmix {
            if ay[i] < TINY {
                continue;
            }
            for j in (i + 1)..nmix {
                if ay[j] < TINY {
                    continue;
                }
                let rz = azion[j] / azion[i];
                let x2 = ay[j] / (ay[i] + ay[j]);
                let x1 = (1.0 - x2).max(0.0);
                if x2 < TINY {
                    continue;
                }
                let x = x2 / rz + (1.0 - 1.0 / rz) * x2.powf(rz);
                let gami = game * azion[i].powf(5.0 / 3.0);
                let mut delta_g = 0.012
                    * (1.0 - 1.0 / (rz * rz))
                    * (x1 + x2 * rz.powf(5.0 / 3.0));
                delta_g *= x / x2 * (1.0 - x).max(0.0) / x1;
                fmix = fmix + gami * (ay[i] * ay[j] * delta_g);
            }
        }
        let z = AutoDiffReal2VarOrder1::zero();
        PcIonContrib {
            f: fmix,
            u: fmix,
            p: fmix / 3.0,
            s: z,
            cv: z,
            pdt: z,
            pdr: fmix / 2.25,
        }
    };

    uint = uint + mix.u * pressi;
    s_tot = s_tot + densi * (mix.u - mix.f);
    press = press + mix.p * pressi;
    cv_tot = cv_tot + densi * mix.cv;
    pdlt = pdlt + pressi * mix.pdt;
    pdlr = pdlr + pressi * mix.pdr;

    let prad_nkt = press_rad / pressi;
    let p_nkt = press / pressi;
    let u_nkt = uint / pressi;
    let s_nk = s_tot / densi;
    let cv = cv_tot / densi;
    let chir = pdlr / press;
    let chit = pdlt / press;

    Ok(Melange9Output {
        prad_nkt,
        dens,
        zmean,
        cmi_mean,
        z2mean,
        gami_mean,
        chi,
        tpt,
        liqsol,
        p_nkt,
        u_nkt,
        s_nk,
        cv,
        chir,
        chit,
    })
}
