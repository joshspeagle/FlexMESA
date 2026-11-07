//! Non-ideal Coulomb free-energy combiner, boundary extrapolation, and
//! smoothed phase transition / latent-heat extraction for the `Skye`
//! equation of state.
//!
//! Reference: Jermyn et al. (2021), ApJ, 913, 72; upstream Fortran MESA
//! `eos/private/skye_coulomb.f90`.

use flexmesa_autodiff::AutoDiffReal2VarOrder3;
use flexmesa_const::{AMU, BOLTZM, FIVE_THIRDS, KERG, PI, QE, RBOHR};

use crate::skye::coulomb_liquid::{
    classical_ocp_liquid_free_energy, excor7_order3,
    liquid_mixing_rule_correction, ocp_liquid_screening_free_energy_correction,
    quantum_ocp_liquid_free_energy_correction, ME_IN_AMU,
};
use crate::skye::coulomb_solid::{
    ocp_solid_anharmonic_free_energy, ocp_solid_harmonic_free_energy,
    ocp_solid_screening_free_energy_correction, solid_mixing_rule_correction,
    SkyeSolidMixingRule,
};

/// Output of `nonideal_corrections` (`skye_coulomb.f90` lines 59-130).
#[derive(Debug, Clone, Copy)]
pub struct SkyeNonidealOutput {
    /// Non-ideal Coulomb free-energy correction in `\mathrm{erg\,g^{-1}}`.
    pub df: AutoDiffReal2VarOrder3,
    /// Smoothed crystallization phase `\phi \in [0, 1]` (`0` liquid, `1` solid).
    pub phase: AutoDiffReal2VarOrder3,
    /// Latent heat derivative `T (\partial S / \partial \ln T)_\rho`.
    pub latent_ddln_t: AutoDiffReal2VarOrder3,
    /// Latent heat derivative `T (\partial S / \partial \ln \rho)_T`.
    pub latent_ddln_rho: AutoDiffReal2VarOrder3,
}

/// Evaluate `ocp_free_energy(LIQSOL, Zion, CMI, GAMI, TPT)`
/// (`skye_coulomb.f90` lines 377-394).
#[must_use]
pub fn ocp_free_energy(
    liqsol: i32,
    gami: AutoDiffReal2VarOrder3,
    tpt: AutoDiffReal2VarOrder3,
) -> AutoDiffReal2VarOrder3 {
    if liqsol == 0 {
        classical_ocp_liquid_free_energy(gami)
            + quantum_ocp_liquid_free_energy_correction(tpt)
    } else {
        ocp_solid_harmonic_free_energy(gami, tpt)
            + ocp_solid_anharmonic_free_energy(gami, tpt)
    }
}

/// Extrapolate the OCP free energy beyond `[min_gamma_for_solid,
/// max_gamma_for_liquid]` by fixing the boundary entropy and internal energy
/// (`skye_coulomb.f90` lines 279-364).
#[must_use]
pub fn extrapolate_free_energy(
    liqsol: i32,
    temp: AutoDiffReal2VarOrder3,
    rs: AutoDiffReal2VarOrder3,
    zion: f64,
    cmi: f64,
    min_gamma_for_solid: f64,
    max_gamma_for_liquid: f64,
) -> AutoDiffReal2VarOrder3 {
    let zi = (zion as i64) as f64;
    let z5_3 = zi.powf(5.0 / 3.0);
    let z7_6 = zi.powf(7.0 / 6.0);
    let gami = (z5_3 * QE * QE) / ((RBOHR * BOLTZM) * temp * rs);
    let cotpt = (3.0 * ME_IN_AMU / cmi).sqrt() / z7_6;
    let tpt = gami * cotpt / rs.sqrt();

    if (liqsol == 0 && gami.val < max_gamma_for_liquid)
        || (liqsol == 1 && gami.val > min_gamma_for_solid)
    {
        ocp_free_energy(liqsol, gami, tpt)
    } else {
        let gamma_boundary = if liqsol == 0 {
            max_gamma_for_liquid
        } else {
            min_gamma_for_solid
        };
        let tb_val = temp.val * gami.val / gamma_boundary;
        let temp_boundary_var1 = AutoDiffReal2VarOrder3::var1(tb_val);
        let g = (z5_3 * QE * QE) / ((RBOHR * BOLTZM) * temp_boundary_var1 * rs);
        let tp = g * cotpt / rs.sqrt();
        let f_bound = ocp_free_energy(liqsol, g, tp);
        let df_dln_t_bound = f_bound.differentiate_1() * temp_boundary_var1;

        let temp_boundary = temp * gami / gamma_boundary;
        let fake_dens = AutoDiffReal2VarOrder3::var2(0.0);
        let df_dln_t = df_dln_t_bound.compose_binary(temp_boundary, fake_dens);
        let f_sub = f_bound.compose_binary(temp_boundary, fake_dens);
        f_sub + (1.0 - temp_boundary / temp) * df_dln_t
    }
}

/// Evaluate `decide_phase` (`skye_coulomb.f90` lines 144-182).
#[must_use]
pub fn decide_phase(
    df_liq: AutoDiffReal2VarOrder3,
    df_sol: AutoDiffReal2VarOrder3,
    kt: AutoDiffReal2VarOrder3,
    temp: AutoDiffReal2VarOrder3,
    rho: AutoDiffReal2VarOrder3,
) -> SkyeNonidealOutput {
    const BLUR_WIDTH: f64 = 100.0;
    let df = df_liq.min(df_sol);
    let blur = kt / BLUR_WIDTH;

    let phase = if df_liq.val < df_sol.val - 20.0 * blur.val {
        AutoDiffReal2VarOrder3::constant(0.0)
    } else if df_sol.val < df_liq.val - 20.0 * blur.val {
        AutoDiffReal2VarOrder3::constant(1.0)
    } else {
        let ex = ((df_liq - df_sol) / blur).exp();
        ex / (ex + 1.0)
    };
    let df_blur = df_liq * (1.0 - phase) + df_sol * phase - df;
    let latent_s = -df_blur.differentiate_1();
    let latent_ddln_t = latent_s.differentiate_1() * temp.pow2();
    let latent_ddln_rho = temp * rho * latent_s.differentiate_2();

    SkyeNonidealOutput {
        df,
        phase,
        latent_ddln_t,
        latent_ddln_rho,
    }
}

#[allow(clippy::too_many_arguments)]
fn nonideal_corrections_phase(
    ay: &[f64],
    azion: &[f64],
    acmi: &[f64],
    min_gamma_for_solid: f64,
    max_gamma_for_liquid: f64,
    solid_mixing_rule: SkyeSolidMixingRule,
    temp: AutoDiffReal2VarOrder3,
    game: AutoDiffReal2VarOrder3,
    rs: AutoDiffReal2VarOrder3,
    liqsol: i32,
    zmean: f64,
    z2mean: f64,
    z52: f64,
    z53: f64,
    z321: f64,
) -> AutoDiffReal2VarOrder3 {
    const TINY: f64 = 1.0e-7;
    let nmix = ay.len().min(azion.len()).min(acmi.len());
    let mut df = AutoDiffReal2VarOrder3::constant(0.0);

    for i in 0..nmix {
        if ay[i] > TINY && azion[i] != 0.0 {
            let mut f = extrapolate_free_energy(
                liqsol,
                temp,
                rs,
                azion[i],
                acmi[i],
                min_gamma_for_solid,
                max_gamma_for_liquid,
            );
            if liqsol == 0 {
                f = f + ocp_liquid_screening_free_energy_correction(
                    azion[i], acmi[i], game, rs,
                );
            } else {
                f = f + ocp_solid_screening_free_energy_correction(
                    azion[i], acmi[i], game, rs,
                );
            }
            df = df + f * ay[i];
        }
    }

    let fmix = if liqsol == 0 {
        liquid_mixing_rule_correction(rs, game, zmean, z2mean, z52, z53, z321)
    } else {
        solid_mixing_rule_correction(solid_mixing_rule, ay, azion, game)
    };
    df + fmix
}

/// Evaluate `nonideal_corrections` (`skye_coulomb.f90` lines 59-130).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn nonideal_corrections(
    ay: &[f64],
    azion: &[f64],
    acmi: &[f64],
    min_gamma_for_solid: f64,
    max_gamma_for_liquid: f64,
    solid_mixing_rule: SkyeSolidMixingRule,
    rho: AutoDiffReal2VarOrder3,
    temp: AutoDiffReal2VarOrder3,
    xnefer: AutoDiffReal2VarOrder3,
    abar: f64,
) -> SkyeNonidealOutput {
    let nmix = ay.len().min(azion.len()).min(acmi.len());
    let mut zmean = 0.0_f64;
    let mut z2mean = 0.0_f64;
    let mut z52 = 0.0_f64;
    let mut z53 = 0.0_f64;
    let mut z321 = 0.0_f64;
    let mut s_mix = 0.0_f64;

    for ix in 0..nmix {
        let y = ay[ix];
        let z = azion[ix];
        zmean += y * z;
        z2mean += y * z * z;
        z53 += y * z.powf(FIVE_THIRDS);
        z52 += y * z.sqrt().powi(5);
        z321 += y * z * (z + 1.0).sqrt().powi(3);
        s_mix -= y * y.ln();
    }

    let dens = xnefer * (RBOHR * RBOHR * RBOHR);
    let rs = (3.0 / ((4.0 * PI) * dens)).powf(1.0 / 3.0);
    let game = (QE * QE) / ((RBOHR * BOLTZM) * temp * rs);

    let f_phase_independent = excor7_order3(rs, game) * zmean - s_mix;

    let df_liq_0 = nonideal_corrections_phase(
        ay,
        azion,
        acmi,
        min_gamma_for_solid,
        max_gamma_for_liquid,
        solid_mixing_rule,
        temp,
        game,
        rs,
        0,
        zmean,
        z2mean,
        z52,
        z53,
        z321,
    );
    let df_sol_0 = nonideal_corrections_phase(
        ay,
        azion,
        acmi,
        min_gamma_for_solid,
        max_gamma_for_liquid,
        solid_mixing_rule,
        temp,
        game,
        rs,
        1,
        zmean,
        z2mean,
        z52,
        z53,
        z321,
    );

    let kt = temp * (KERG / (abar * AMU));
    let df_liq = (df_liq_0 + f_phase_independent) * kt;
    let df_sol = (df_sol_0 + f_phase_independent) * kt;

    decide_phase(df_liq, df_sol, kt, temp, rho)
}
