// SPDX-License-Identifier: LGPL-3.0-or-later
//! White-dwarf crystallization chemical phase separation ported from MESA
//! `26.4.1` `star/private/phase_separation.f90` (Blouin et al. 2021a,b C/O and
//! O/Ne phase diagrams, constant-`Pgas` EOS updates with
//! `Skye_use_ion_offsets = .false.`, outward mixing by Ledoux/Brunt-B or
//! mean-molecular-weight stability, and `eps_phase_separation` energy release).
//!
//! Note: MESA `26.4.1` `star/private/phase_separation.f90` (400 lines) provides
//! only the two-component Blouin `'CO'` and `'ONe'` diagrams; three-component
//! Medin-Cumming phase tables are absent from upstream `26.4.1`.

use flexmesa_chem::{get_composition_info, NuclideCatalog};
use flexmesa_const::{
    AVO, CLIGHT, CRAD, FOUR_PI, KERG, LN10, MixingType, QE,
};
use flexmesa_eos::{
    eos_dt_get, eos_dt_get_rho, I_CHI_RHO, I_CHI_T, I_CP, I_CV, I_ETA,
    I_FRAC_PC, I_GAMMA1, I_GAMMA3, I_GRAD_AD, I_LATENT_DDLNRHO,
    I_LATENT_DDLNT, I_LN_E, I_LN_FREE_E, I_LN_PGAS, I_LN_S, I_MU, I_PHASE,
};
use flexmesa_kap::get_kap_results;

use crate::job_controls::StarError;
use crate::micro::builtin_helm_table;
use crate::model::StarModel;
use crate::star_ad::{StarAdExt, StarAdOrder1};

/// EOS phase boundary (`0.9`) separating liquid (`<= 0.9`) and solid (`> 0.9`)
/// zones (`star/private/phase_separation.f90:34`).
pub const EOS_PHASE_BOUNDARY: f64 = 0.9;

/// Constant-`Pgas` density inversion tolerances matching `micro.f90:334`.
const LOGRHO_TOL: f64 = 1.0e-8;
const LOGPGAS_TOL: f64 = 1.0e-8;

/// Evaluates the Blouin et al. (2021a) oxygen mass-fraction enrichment
/// $\Delta X_{\mathrm{O}}$ upon crystallization of a C/O mixture
/// (`star/private/phase_separation.f90:269-299`).
#[must_use]
pub fn blouin_delta_xo(x_in: f64) -> f64 {
    let mut xo = (x_in / 16.0) / (x_in / 16.0 + (1.0 - x_in) / 12.0);

    let a0 = 0.0;
    let a1 = -0.311540;
    let a2 = 2.114743;
    let a3 = -1.661095;
    let a4 = -1.406005;
    let a5 = 1.263897;

    let dxo = a0
        + a1 * xo
        + a2 * xo * xo
        + a3 * xo * xo * xo
        + a4 * xo * xo * xo * xo
        + a5 * xo * xo * xo * xo * xo;

    xo += dxo;
    let x_new = 16.0 * xo / (16.0 * xo + 12.0 * (1.0 - xo));
    x_new - x_in
}

/// Evaluates the Blouin et al. (2021b) neon mass-fraction enrichment
/// $\Delta X_{\mathrm{Ne}}$ upon crystallization of an O/Ne mixture
/// (`star/private/phase_separation.f90:301-331`).
#[must_use]
pub fn blouin_delta_xne(x_in: f64) -> f64 {
    let mut xne = (x_in / 20.0) / (x_in / 20.0 + (1.0 - x_in) / 16.0);

    let a0 = 0.0;
    let a1 = -0.120299;
    let a2 = 1.304399;
    let a3 = -1.722625;
    let a4 = 0.393996;
    let a5 = 0.144529;

    let dxne = a0
        + a1 * xne
        + a2 * xne * xne
        + a3 * xne * xne * xne
        + a4 * xne * xne * xne * xne
        + a5 * xne * xne * xne * xne * xne;

    xne += dxne;
    let x_new = 20.0 * xne / (20.0 * xne + 16.0 * (1.0 - xne));
    x_new - x_in
}

/// Evaluates the solid-phase heavy-species mass-fraction enrichment
/// $\Delta X_{\mathrm{heavy}}$ for either `'CO'` (`is_one_mixture = false`)
/// or `'ONe'` (`is_one_mixture = true`), matching
/// `star/private/phase_separation.f90:269-331`.
#[must_use]
pub fn eval_phase_separation_shift(x_heavy: f64, is_one_mixture: bool) -> f64 {
    if is_one_mixture {
        blouin_delta_xne(x_heavy)
    } else {
        blouin_delta_xo(x_heavy)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PhaseComponents {
    Co,
    ONe,
}

/// Executes one crystallization phase-separation step on `model` over timestep
/// `dt` (s), matching `star/private/phase_separation.f90:38-267`.
///
/// Updates `model.xa`, `model.lnd`, `model.rho`, thermodynamic/opacity arrays,
/// `model.aux.crystal_core_boundary_mass`, `model.aux.phase_sep_mixing_mass`,
/// and `model.aux.eps_phase_separation`.
///
/// # Errors
///
/// Returns [`StarError::InvalidInput`] for an unrecognized
/// `phase_separation_option`, missing required CO/ONe isotopes, or any of the
/// upstream out-of-bounds/uninitialized boundary conditions tracked in `#271`,
/// and [`StarError::MicrophysicsError`] if constant-`Pgas` EOS, opacity, or
/// Brunt-B evaluation fails.
pub fn do_phase_separation_step(
    model: &mut StarModel,
    dt: f64,
) -> Result<(), StarError> {
    let opt = model.controls.phase_separation_option.trim();
    let components = match opt {
        "CO" => PhaseComponents::Co,
        "ONe" => PhaseComponents::ONe,
        other => {
            return Err(StarError::InvalidInput(format!(
                "invalid phase_separation_option '{other}' (expected 'CO' or 'ONe')"
            )));
        }
    };
    do_2component_phase_separation(model, dt, components)
}

#[allow(clippy::too_many_lines)]
fn do_2component_phase_separation(
    model: &mut StarModel,
    dt: f64,
    components: PhaseComponents,
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;

    model.aux.phase_sep_mixing_mass = -1.0;
    model.aux.eps_phase_separation.resize(nz, 0.0);
    model.aux.eps_phase_separation.fill(0.0);

    if nz < 2 || species < 2 {
        return Ok(());
    }

    if model.eos_res[nz - 1][I_PHASE] < EOS_PHASE_BOUNDARY {
        model.aux.crystal_core_boundary_mass = 0.0;
        return Ok(());
    }

    let catalog = NuclideCatalog::builtin_ref();
    let find_iso = |name: &str| -> Option<usize> {
        catalog
            .get_by_name(name)
            .and_then(|iso| model.net_iso.get(iso.chem_id).copied().flatten())
    };

    // Guard #271: require only the two isotopes for the active component pair
    // rather than unconditionally indexing all three (c12, o16, ne20).
    let (idx_light, idx_heavy) = match components {
        PhaseComponents::Co => {
            let ic12 = find_iso("c12").ok_or_else(|| {
                StarError::InvalidInput(
                    "phase_separation option 'CO' requires c12 in the network (#271)"
                        .into(),
                )
            })?;
            let io16 = find_iso("o16").ok_or_else(|| {
                StarError::InvalidInput(
                    "phase_separation option 'CO' requires o16 in the network (#271)"
                        .into(),
                )
            })?;
            (ic12, io16)
        }
        PhaseComponents::ONe => {
            let io16 = find_iso("o16").ok_or_else(|| {
                StarError::InvalidInput(
                    "phase_separation option 'ONe' requires o16 in the network (#271)"
                        .into(),
                )
            })?;
            let ine20 = find_iso("ne20").ok_or_else(|| {
                StarError::InvalidInput(
                    "phase_separation option 'ONe' requires ne20 in the network (#271)"
                        .into(),
                )
            })?;
            (io16, ine20)
        }
    };

    // Locate liquid-to-solid crystallization front k_bound
    // (phase_separation.f90:79-85; guarded against phase(0) and k_bound == -1 per #271).
    let mut k_bound: Option<usize> = None;
    for k in (1..nz).rev() {
        if model.eos_res[k - 1][I_PHASE] <= EOS_PHASE_BOUNDARY
            && model.eos_res[k][I_PHASE] > EOS_PHASE_BOUNDARY
        {
            k_bound = Some(k);
            break;
        }
    }
    let Some(k_bound) = k_bound else {
        return Err(StarError::InvalidInput(
            "phase_separation reached upstream OOB boundary condition \
             (no liquid-solid phase transition found or surface zone is \
             crystallized; tracked under #271)"
                .into(),
        ));
    };

    let x_light = model.xa[k_bound * species + idx_light];
    let x_heavy = model.xa[k_bound * species + idx_heavy];
    match components {
        PhaseComponents::Co => {
            if x_heavy + x_light < 0.9 {
                return Ok(());
            }
        }
        PhaseComponents::ONe => {
            if x_heavy + x_light < 0.8 {
                return Ok(());
            }
        }
    }

    // Locate starting zone kstart just outside prior crystal_core_boundary_mass
    // (phase_separation.f90:100-106; guarded against uninitialized kstart per #271).
    let pad = model.controls.mix_wind.min_dq * model.m[0] * 0.5;
    let mut kstart: Option<usize> = None;
    for k in (0..nz).rev() {
        if model.m[k] > model.aux.crystal_core_boundary_mass + pad {
            kstart = Some(k);
            break;
        }
    }
    let Some(kstart) = kstart else {
        return Err(StarError::InvalidInput(
            "phase_separation could not locate crystallization start zone \
             kstart above crystal_core_boundary_mass (#271)"
                .into(),
        ));
    };

    // Evaluate phase-separation energy with Skye_use_ion_offsets temporarily
    // disabled (phase_separation.f90:110-139), restoring unconditionally.
    let save_skye_ion_offsets = model.eos_config.skye.skye_use_ion_offsets;
    model.eos_config.skye.skye_use_ion_offsets = false;

    let result = (|| -> Result<(), StarError> {
        let mut unsmoothed_brunt_b = vec![0.0_f64; nz];
        let mut grada_face = vec![0.0_f64; nz];

        update_model_range(
            model,
            0,
            nz - 1,
            false,
            &mut unsmoothed_brunt_b,
            &mut grada_face,
        )?;
        let energy_before = model.energy.clone();

        for k in (0..=kstart).rev() {
            if model.eos_res[k][I_PHASE] <= EOS_PHASE_BOUNDARY {
                model.aux.crystal_core_boundary_mass = if k + 1 < nz {
                    model.m[k + 1]
                } else {
                    model.m_center
                };
                break;
            }
            if k == 0 {
                return Err(StarError::InvalidInput(
                    "phase_separation reached surface zone k=0 in move_one_zone (#271)"
                        .into(),
                ));
            }

            move_one_zone(
                model,
                k,
                components,
                idx_light,
                idx_heavy,
                &mut unsmoothed_brunt_b,
                &mut grada_face,
            )?;
            mix_outward(
                model,
                k - 1,
                0,
                &mut unsmoothed_brunt_b,
                &mut grada_face,
            )?;
        }

        update_model_range(
            model,
            0,
            nz - 1,
            false,
            &mut unsmoothed_brunt_b,
            &mut grada_face,
        )?;

        if dt > 0.0 && dt.is_finite() {
            for k in 0..nz {
                model.aux.eps_phase_separation[k] =
                    (energy_before[k] - model.energy[k]) / dt;
            }
        }
        Ok(())
    })();

    model.eos_config.skye.skye_use_ion_offsets = save_skye_ion_offsets;
    result
}

#[allow(clippy::too_many_arguments)]
fn move_one_zone(
    model: &mut StarModel,
    k: usize,
    components: PhaseComponents,
    idx_light: usize,
    idx_heavy: usize,
    unsmoothed_brunt_b: &mut [f64],
    grada_face: &mut [f64],
) -> Result<(), StarError> {
    let species = model.species;
    let nz = model.nz;

    let x_heavy_raw = model.xa[k * species + idx_heavy];
    let x_light_raw = model.xa[k * species + idx_light];
    let xfac = x_heavy_raw + x_light_raw;
    let x_heavy = x_heavy_raw / xfac;
    let x_light = x_light_raw / xfac;

    let dx_heavy = match components {
        PhaseComponents::Co => blouin_delta_xo(x_heavy),
        PhaseComponents::ONe => blouin_delta_xne(x_heavy),
    };

    model.xa[k * species + idx_heavy] = xfac * (x_heavy + dx_heavy);
    model.xa[k * species + idx_light] = xfac * (x_light - dx_heavy);

    let dq_ratio = model.dq[k] / model.dq[k - 1];
    let x_light_above = model.xa[(k - 1) * species + idx_light];
    let x_heavy_above = model.xa[(k - 1) * species + idx_heavy];
    model.xa[(k - 1) * species + idx_light] =
        x_light_above + xfac * dx_heavy * dq_ratio;
    model.xa[(k - 1) * species + idx_heavy] =
        x_heavy_above - xfac * dx_heavy * dq_ratio;

    update_model_range(
        model,
        k - 1,
        nz - 1,
        true,
        unsmoothed_brunt_b,
        grada_face,
    )
}

fn mix_outward(
    model: &mut StarModel,
    kbot: usize,
    min_mix_zones: usize,
    unsmoothed_brunt_b: &mut [f64],
    grada_face: &mut [f64],
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let use_brunt = model.controls.diffusion.phase_separation_mixing_use_brunt;
    let mut avg_xa = vec![0.0_f64; species];
    let mut last_ktop = kbot.saturating_sub(min_mix_zones);

    if kbot < min_mix_zones {
        return Err(StarError::InvalidInput(
            "phase_separation mix_outward kbot < min_mix_zones (#271)".into(),
        ));
    }

    for k in (0..=(kbot - min_mix_zones)).rev() {
        let ktop = k;
        last_ktop = ktop;

        if model.m[ktop] > model.aux.phase_sep_mixing_mass {
            model.aux.phase_sep_mixing_mass = model.m[ktop];
        }

        let mut mass = 0.0;
        for kk in ktop..=kbot {
            mass += model.dm[kk];
        }
        for l in 0..species {
            let mut acc = 0.0;
            for kk in ktop..=kbot {
                acc += model.dm[kk] * model.xa[kk * species + l];
            }
            avg_xa[l] = acc / mass;
        }
        for kk in ktop..=kbot {
            for l in 0..species {
                model.xa[kk * species + l] = avg_xa[l];
            }
        }

        // Guard #271: upstream calls `update_model_(s, ktop - 1, ktop + 1, ...)`
        // and accesses `mu(ktop - 1)`, which is out-of-bounds if `ktop == 0`.
        if ktop == 0 {
            return Err(StarError::InvalidInput(
                "phase_separation mix_outward reached surface zone ktop == 0 \
                 (upstream OOB ktop - 1 = 0; tracked under #271)"
                    .into(),
            ));
        }

        update_model_range(
            model,
            ktop - 1,
            (ktop + 1).min(nz - 1),
            use_brunt,
            unsmoothed_brunt_b,
            grada_face,
        )?;

        if use_brunt {
            let b_term = unsmoothed_brunt_b[ktop];
            let grada = grada_face[ktop];
            let gradr = model.gradr[ktop];
            if b_term + grada - gradr > 0.0 {
                break;
            }
        } else if model.mu[ktop] >= model.mu[ktop - 1] {
            break;
        }
    }

    update_model_range(
        model,
        last_ktop,
        (kbot + 1).min(nz - 1),
        true,
        unsmoothed_brunt_b,
        grada_face,
    )
}

/// Refreshes constant-`Pgas` EOS (`fix_Pgas = .true.`), opacities, optional
/// unsmoothed Brassard Brunt-B (`brunt.f90:231-313`), and interior face
/// `grada_face`/`gradr` over zones `kc_t..=kc_b`
/// (`star/private/phase_separation.f90:333-398`).
#[allow(clippy::too_many_lines)]
fn update_model_range(
    model: &mut StarModel,
    kc_t: usize,
    kc_b: usize,
    do_brunt: bool,
    unsmoothed_brunt_b: &mut [f64],
    grada_face: &mut [f64],
) -> Result<(), StarError> {
    let nz = model.nz;
    let species = model.species;
    let kc_hi = kc_b.min(nz - 1);
    if kc_t > kc_hi {
        return Ok(());
    }

    let ht = builtin_helm_table()?;
    let catalog = NuclideCatalog::builtin_ref();
    let iso_z: Vec<usize> = model
        .chem_id
        .iter()
        .map(|&cid| catalog.get_by_id(cid).map_or(0, |n| n.z))
        .collect();

    // 1. Constant-(Pgas, T) EOS solve over kc_t..=kc_hi (micro.f90:321-385).
    for k in kc_t..=kc_hi {
        let xa_k = &model.xa[k * species..(k + 1) * species];
        let comp = get_composition_info(catalog, &model.chem_id, xa_k, false)
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "phase_separation composition error at zone {k}: {e}"
                ))
            })?;
        model.abar[k] = comp.abar;
        model.zbar[k] = comp.zbar;
        model.z2bar[k] = comp.z2bar;
        model.z53bar[k] = comp.z53bar;
        model.ye[k] = comp.ye;
        model.xh_mass[k] = comp.xh.clamp(0.0, 1.0);
        model.yhe_mass[k] = comp.xhe.clamp(0.0, 1.0);
        model.z_mass[k] = comp.xz.clamp(0.0, 1.0);

        let log_t = model.ln_t[k] / LN10;
        let log_rho_guess = model.lnd[k] / LN10;
        let target_ln_pgas = model.ln_pgas[k];

        let inv = eos_dt_get_rho(
            ht,
            &model.chem_id,
            xa_k,
            log_t,
            I_LN_PGAS as isize,
            target_ln_pgas,
            log_rho_guess,
            LOGRHO_TOL,
            LOGPGAS_TOL,
            20,
            &model.eos_config,
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "phase_separation constant-Pgas EOS failed at zone {k}: {e}"
            ))
        })?;

        let lnd_k = inv.log10_result * LN10;
        let rho_k = inv.linear_result;
        let t_k = model.t[k];
        model.lnd[k] = lnd_k;
        model.rho[k] = rho_k;
        let out = inv.eos;

        let ln_pgas = out.res[I_LN_PGAS];
        let pgas = flexmesa_math::exp(ln_pgas);
        let prad = CRAD * flexmesa_math::pow4(t_k) / 3.0;
        let peos = pgas + prad;
        let ln_peos = flexmesa_math::log(peos);

        let d_lnpgas_dlnd = out.d_dlnd[I_LN_PGAS];
        let d_lnpgas_dlnt = out.d_dln_t[I_LN_PGAS];
        let alfa_rad = prad / peos;
        let beta_gas = pgas / peos;
        let d_lnpeos_dlnd = beta_gas * d_lnpgas_dlnd;
        let d_lnpeos_dlnt = beta_gas * d_lnpgas_dlnt + 4.0 * alfa_rad;

        let ln_e = out.res[I_LN_E];
        let energy = flexmesa_math::exp(ln_e);
        let ln_s = out.res[I_LN_S];
        let entropy = flexmesa_math::exp(ln_s);
        let gamma1 = out.res[I_GAMMA1];
        let csound = (gamma1 * peos / rho_k).max(0.0).sqrt();

        let a_e = flexmesa_math::pow(
            3.0 / (FOUR_PI * AVO * rho_k * model.zbar[k].max(1.0e-10)
                / model.abar[k]),
            1.0 / 3.0,
        );
        let gam = model.z53bar[k] * QE * QE / (a_e * KERG * t_k);

        model.pgas[k] = pgas;
        model.ln_pgas[k] = ln_pgas;
        model.prad[k] = prad;
        model.peos[k] = peos;
        model.ln_peos[k] = ln_peos;
        model.energy[k] = energy;
        model.ln_e[k] = ln_e;
        model.entropy[k] = entropy;
        model.ln_s[k] = ln_s;
        model.grada[k] = out.res[I_GRAD_AD];
        model.cv[k] = out.res[I_CV];
        model.cp[k] = out.res[I_CP];
        model.chi_rho[k] = out.res[I_CHI_RHO];
        model.chi_t[k] = out.res[I_CHI_T];
        model.gamma1[k] = gamma1;
        model.gamma3[k] = out.res[I_GAMMA3];
        model.eta[k] = out.res[I_ETA];
        model.gam[k] = gam;
        model.mu[k] = out.res[I_MU];
        model.lnfree_e[k] = out.res[I_LN_FREE_E];
        model.csound[k] = csound;

        model.aux.eos_frac_pc[k] = out.res[I_FRAC_PC];
        model.aux.latent_ddlnt[k] = out.res[I_LATENT_DDLNT];
        model.aux.latent_ddlnrho[k] = out.res[I_LATENT_DDLNRHO];

        model.ln_peos_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
            ln_peos,
            d_lnpeos_dlnd,
            d_lnpeos_dlnt,
        );
        model.ln_e_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
            ln_e,
            out.d_dlnd[I_LN_E],
            out.d_dln_t[I_LN_E],
        );
        model.ln_s_ad[k] = StarAdOrder1::from_dlnd_dlnt_00(
            ln_s,
            out.d_dlnd[I_LN_S],
            out.d_dln_t[I_LN_S],
        );
        model.eos_res[k] = out.res;
        model.eos_d_dlnd[k] = out.d_dlnd;
        model.eos_d_dlnt[k] = out.d_dln_t;
    }

    // 2. Opacities over kc_t..=kc_hi (phase_separation.f90:365-366).
    let kap_factor = model.controls.opacity_factor;
    for k in kc_t..=kc_hi {
        let xa_k = &model.xa[k * species..(k + 1) * species];
        let mut xc = 0.0;
        let mut xn = 0.0;
        let mut xo = 0.0;
        let mut xne = 0.0;
        for (&z_iso, &xj) in iso_z.iter().zip(xa_k.iter()) {
            match z_iso {
                6 => xc += xj,
                7 => xn += xj,
                8 => xo += xj,
                10 => xne += xj,
                _ => {}
            }
        }
        let out = get_kap_results(
            &model.kap_config,
            model.zbar[k],
            model.xh_mass[k],
            model.z_mass[k],
            xc,
            xn,
            xo,
            xne,
            flexmesa_math::log10(model.rho[k]),
            flexmesa_math::log10(model.t[k]),
            model.lnfree_e[k],
            model.eos_d_dlnd[k][I_LN_FREE_E],
            model.eos_d_dlnt[k][I_LN_FREE_E],
            model.eta[k],
            model.eos_d_dlnd[k][I_ETA],
            model.eos_d_dlnt[k][I_ETA],
        )
        .map_err(|e| {
            StarError::MicrophysicsError(format!(
                "phase_separation opacity failed at zone {k}: {e}"
            ))
        })?;
        model.opacity[k] = out.kap * kap_factor;
        model.d_opacity_dlnd[k] = out.kap * out.dlnkap_dln_rho * kap_factor;
        model.d_opacity_dlnt[k] = out.kap * out.dlnkap_dln_t * kap_factor;
    }

    // 3. Unsmoothed Brunt-B (Brassard finite-difference EOS at face (rho, T),
    //    matching brunt.f90:231-313).
    if do_brunt {
        for k in kc_t..=kc_hi {
            if k == 0 {
                unsmoothed_brunt_b[0] = 0.0;
                continue;
            }
            let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
            let beta = 1.0 - alfa;
            let t_face = alfa * model.t[k] + beta * model.t[k - 1];
            let rho_face = alfa * model.rho[k] + beta * model.rho[k - 1];
            let chi_t_face = alfa * model.chi_t[k] + beta * model.chi_t[k - 1];
            let chi_rho_face =
                alfa * model.chi_rho[k] + beta * model.chi_rho[k - 1];
            let prad_face = CRAD * t_face * t_face * t_face * t_face / 3.0;

            let xa_k = &model.xa[k * species..(k + 1) * species];
            let xa_km1 = &model.xa[(k - 1) * species..k * species];

            let out1 = eos_dt_get(
                ht,
                rho_face,
                t_face,
                &model.chem_id,
                xa_k,
                &model.eos_config,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "phase_separation brunt_B EOS(k) failed at face {k}: {e}"
                ))
            })?;
            let lnp1 =
                flexmesa_math::log(prad_face + flexmesa_math::exp(out1.res[I_LN_PGAS]));

            let out2 = eos_dt_get(
                ht,
                rho_face,
                t_face,
                &model.chem_id,
                xa_km1,
                &model.eos_config,
            )
            .map_err(|e| {
                StarError::MicrophysicsError(format!(
                    "phase_separation brunt_B EOS(k-1) failed at face {k}: {e}"
                ))
            })?;
            let lnp2 =
                flexmesa_math::log(prad_face + flexmesa_math::exp(out2.res[I_LN_PGAS]));

            let mut delta_lnp = model.ln_peos[k - 1] - model.ln_peos[k];
            if delta_lnp > -1.0e-50 {
                let p_point = alfa * model.peos[k] + beta * model.peos[k - 1];
                let dlnp_dm = -model.cgrav[k] * model.m[k]
                    / (FOUR_PI * flexmesa_math::pow4(model.r[k]) * p_point);
                delta_lnp = dlnp_dm * model.dm_bar[k];
            }

            let mut b_k = (lnp1 - lnp2) / delta_lnp / chi_t_face;
            if model.controls.use_mass_corrections
                && model.aux.mass_correction.len() == nz
            {
                let delta_ln_mbar = flexmesa_math::log(
                    model.aux.mass_correction[k - 1],
                ) - flexmesa_math::log(model.aux.mass_correction[k]);
                b_k -= chi_rho_face * delta_ln_mbar / delta_lnp / chi_t_face;
            }
            if !b_k.is_finite() {
                return Err(StarError::MicrophysicsError(format!(
                    "non-finite unsmoothed_brunt_B at face {k}"
                )));
            }
            unsmoothed_brunt_b[k] = b_k;
            model.brunt_b[k] = b_k;
        }
    }

    // 4. Face grada_face and gradr over interior faces (kc_t + 1)..=kc_hi
    //    (phase_separation.f90:387-390, hydro_vars.rs:140-187).
    for k in (kc_t + 1)..=kc_hi {
        let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
        let beta = 1.0 - alfa;
        let p_f = alfa * model.peos[k] + beta * model.peos[k - 1];
        let prad_f = alfa * model.prad[k] + beta * model.prad[k - 1];
        let kap_f = alfa * model.opacity[k] + beta * model.opacity[k - 1];
        let grada_f = alfa * model.grada[k] + beta * model.grada[k - 1];
        let (fp_k, ft_k) = if model.controls.rotation_flag
            && model.controls.use_gravity_rotation_correction
        {
            let fp = model.aux.fp_rot.get(k).copied().unwrap_or(1.0);
            let ft = model.aux.ft_rot.get(k).copied().unwrap_or(1.0);
            (
                if fp > 0.0 { fp } else { 1.0 },
                if ft > 0.0 { ft } else { 1.0 },
            )
        } else {
            (1.0, 1.0)
        };
        let gm_k = model.cgrav[k] * model.m_grav[k];
        let gradr_f = (p_f * kap_f * model.lum[k]
            / (4.0 * FOUR_PI * CLIGHT * gm_k * prad_f))
            * (ft_k / fp_k);
        grada_face[k] = grada_f;
        model.gradr[k] = gradr_f;
    }

    Ok(())
}

/// Applies crystallized and phase-separation mixing classifications and zeroes
/// convective mixing inside solidified/phase-separated shells, matching
/// `star/private/turb_info.f90:143-159`.
pub fn apply_phase_sep_mixing_types(model: &mut StarModel) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    let crystal_pad = model.controls.mix_wind.min_dq * model.m[0] * 0.5;
    let m_cryst = model.aux.crystal_core_boundary_mass;
    let m_mix = model.aux.phase_sep_mixing_mass;

    for k in 0..nz {
        let phase_k = model.eos_res.get(k).map_or(0.0, |r| r[I_PHASE]);
        let is_solid = (phase_k > 0.5 && model.mu[k] > 1.7)
            || (m_cryst > 0.0 && m_cryst + crystal_pad > model.m[k]);
        let is_phase_sep = m_mix > 0.0 && model.m[k] <= m_mix;

        if is_solid || is_phase_sep {
            model.gradt[k] = model.gradr[k];
            model.gradt_ad[k] = model.gradr_ad[k];
            model.mlt_vc[k] = 0.0;
            model.mlt_vc_ad[k] = StarAdOrder1::ZERO;
            model.mlt_d[k] = 0.0;
            model.mlt_d_ad[k] = StarAdOrder1::ZERO;
            model.mlt_cdc[k] = 0.0;
            model.mlt_cdc_ad[k] = StarAdOrder1::ZERO;
            model.mlt_gamma[k] = 0.0;
            model.d_mix[k] = 0.0;
            model.conv_vel[k] = 0.0;
            model.sig[k] = 0.0;
            model.mixing_type[k] = if is_solid {
                MixingType::Crystallized
            } else {
                MixingType::PhaseSeparationMixing
            };
        }
    }
}

/// Compatibility entry point executing real MESA `26.4.1` two-component phase
/// separation for `'CO'` (`is_one_mixture = false`) or `'ONe'`
/// (`is_one_mixture = true`) and returning `eps_phase_separation`.
///
/// # Errors
///
/// Propagates any [`StarError`] from [`do_phase_separation_step`].
pub fn do_phase_separation(
    model: &mut StarModel,
    dt: f64,
    _gamma_crystallization: f64,
    is_one_mixture: bool,
) -> Result<Vec<f64>, StarError> {
    let saved_opt = model.controls.phase_separation_option.clone();
    model.controls.phase_separation_option = if is_one_mixture {
        String::from("ONe")
    } else {
        String::from("CO")
    };
    let res = do_phase_separation_step(model, dt);
    model.controls.phase_separation_option = saved_opt;
    res?;
    Ok(model.aux.eps_phase_separation.clone())
}
