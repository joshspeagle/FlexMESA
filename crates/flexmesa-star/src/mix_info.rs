// SPDX-License-Identifier: LGPL-3.0-or-later
//! Convective boundary detection, mixing glitch removal, face-centered
//! Lagrangian diffusion coupling factors ($\sigma_k$), and conservative
//! species mixing rates (`mix_info.f90`).
//!
//! # Sub-Cell Convective Boundaries (`cz_bdy_dq` and `ConvBoundary`)
//! For each cell $k \in \{0, \dots, \mathrm{nz}-2\}$ bounded by outer face $k$
//! and inner face $k+1$ where `mixing_type[k] != mixing_type[k + 1]`, the
//! fractional mass offset from outer face $k$ to the stability crossing
//! $\nabla_L - \nabla_{\mathrm{rad}} = 0$ is interpolated linearly:
//! $$\Delta q_{\mathrm{bdy},k} =
//!   \frac{-\delta g_k}{\delta g_{k+1} - \delta g_k}\,\Delta q_k
//!   \in [0, \Delta q_k],
//!   \quad q_{\mathrm{bdy}} = q_k - \Delta q_{\mathrm{bdy},k}.$$
//!
//! # Discrete Species Diffusion Stencil (`eval_dxdt_mix`)
//! With Lagrangian face coupling factor
//! $$\sigma_k = \mathrm{mix\_factor}\,
//!   \frac{(4\pi r_k^2 \rho_{\mathrm{face},k})^2 D_{\mathrm{mix},k}}
//!        {\overline{\Delta m}_k}
//!   \quad [\mathrm{g\,s^{-1}}]$$
//! ($\sigma_0 = 0$ at the surface and $\sigma_{\mathrm{nz}} = 0$ at the center),
//! the diffusive rate of change of species $j$ in cell $k$ is:
//! $$\left(\frac{dX_{j,k}}{dt}\right)_{\mathrm{mix}} =
//!   \frac{\sigma_k (X_{j,k-1} - X_{j,k})
//!         - \sigma_{k+1} (X_{j,k} - X_{j,k+1})}{\Delta m_k},$$
//! which conserves
//! $\sum_{k=0}^{\mathrm{nz}-1} \Delta m_k (dX_{j,k}/dt)_{\mathrm{mix}} = 0$
//! to machine precision.

use flexmesa_const::{MixingType, FOUR_PI};

use crate::model::StarModel;

/// Sub-cell convective boundary descriptor (`locate_convection_boundaries`
/// in `star/private/mix_info.f90:528-784`).
#[derive(Debug, Clone, PartialEq)]
pub struct ConvBoundary {
    /// True if this boundary is the outer (top) edge of a convection zone;
    /// false if it is the inner (bottom) edge.
    pub top_conv_bdy: bool,
    /// 0-based face index nearest to the boundary inside the convection zone
    /// (`k + 1` for a top boundary in cell `k`, `k` for a bottom boundary in
    /// cell `k`).
    pub conv_bdy_loc: usize,
    /// 0-based cell index containing the sub-cell boundary (`k_cb`).
    pub cell_index: usize,
    /// Interior mass fraction $q_{\mathrm{cb}} \in (0, 1]$ at the boundary.
    pub conv_bdy_q: f64,
    /// Inward fractional mass offset
    /// $\Delta q_{\mathrm{bdy}} \in [0, \Delta q_k]$ from outer face
    /// `cell_index` to the zero crossing.
    pub cz_bdy_dq: f64,
    /// Sub-cell boundary radius $r_{\mathrm{cb}}$ in cm (`eval_conv_bdy_r`).
    pub r_cb: f64,
    /// Sub-cell pressure scale height $H_{P,\mathrm{cb}}$ in cm
    /// (`eval_conv_bdy_Hp`).
    pub hp_cb: f64,
}

/// Computes the face-interpolated density $\rho_{\mathrm{face},k}$ at interior
/// face $k \in \{1, \dots, \mathrm{nz}-1\}$ (`mix_info.f90:116-122`).
#[must_use]
#[inline]
pub fn face_density(model: &StarModel, k: usize) -> f64 {
    if k >= model.nz {
        model.rho.first().copied().unwrap_or(0.0)
    } else {
        model.interp_val_to_pt(&model.rho, k)
    }
}

/// Computes the geometric factor $(4\pi r_k^2 \rho_{\mathrm{face},k})^2$ at
/// face $k$ (`cdc_factor` in `mix_info.f90:118-122`).
#[must_use]
#[inline]
pub fn cdc_factor(model: &StarModel, k: usize) -> f64 {
    if k == 0 || k >= model.nz {
        1.0
    } else {
        let rho_f = face_density(model, k);
        let area_rho = FOUR_PI * model.r[k] * model.r[k] * rho_f;
        area_rho * area_rho
    }
}

/// Removes single-zone mixing glitches, tiny diffusion coefficients, narrow
/// radiative gaps, thermohaline dropouts, and embedded semiconvective zones
/// (`remove_mixing_glitches` in `mix_info.f90:157-180, 965-1348`).
pub fn clean_mixing_glitches(
    model: &mut StarModel,
    clip_d_limit: f64,
    min_convective_gap_hp: f64,
) {
    let nz = model.nz;
    if nz < 3 {
        return;
    }

    // 1. Remove tiny mixing below clip_d_limit (`remove_tiny_mixing`).
    let d_floor = clip_d_limit.max(model.controls.min_d_mix);
    if d_floor > 0.0 {
        for k in 1..nz {
            if model.d_mix[k] > 0.0 && model.d_mix[k] < d_floor {
                set_face_radiative(model, k);
            }
        }
    }

    // 2. Remove mixing singletons (`remove_mixing_singletons`).
    for k in 1..(nz - 1) {
        let prev_mix = model.d_mix[k - 1] > 0.0;
        let curr_mix = model.d_mix[k] > 0.0;
        let next_mix = model.d_mix[k + 1] > 0.0;
        if !curr_mix && prev_mix && next_mix {
            let cdc_m1 = cdc_factor(model, k - 1) * model.d_mix[k - 1];
            let cdc_p1 = cdc_factor(model, k + 1) * model.d_mix[k + 1];
            let cdc_k = 0.5 * (cdc_m1 + cdc_p1);
            let fac_k = cdc_factor(model, k).max(1.0e-99);
            model.mlt_cdc[k] = cdc_k;
            model.d_mix[k] = cdc_k / fac_k;
            model.conv_vel[k] =
                0.5 * (model.conv_vel[k - 1] + model.conv_vel[k + 1]);
            let code = model.mixing_type[k - 1]
                .code()
                .max(model.mixing_type[k + 1].code());
            model.mixing_type[k] = MixingType::from_code(code)
                .unwrap_or(MixingType::ConvectiveMixing);
        } else if curr_mix && !prev_mix && !next_mix {
            set_face_radiative(model, k);
        }
    }

    // 3. Close narrow radiative gaps between convective zones (`close_gaps`).
    if min_convective_gap_hp > 0.0 {
        bridge_zone_gaps(
            model,
            MixingType::ConvectiveMixing,
            MixingType::ConvectiveMixing,
            min_convective_gap_hp,
        );
        // 4. Bridge narrow radiative dropouts inside thermohaline regions
        // (`remove_thermohaline_dropouts`).
        bridge_zone_gaps(
            model,
            MixingType::ThermohalineMixing,
            MixingType::ThermohalineMixing,
            min_convective_gap_hp,
        );
    }

    // 5. Convert embedded semiconvection bounded by convection into convection
    // (`remove_embedded_semiconvection`).
    let mut k = 1;
    while k + 2 < nz {
        if model.mixing_type[k] == MixingType::ConvectiveMixing
            && model.mixing_type[k + 1] == MixingType::SemiconvectiveMixing
        {
            let mut k_bot = k + 1;
            while k_bot < nz
                && model.mixing_type[k_bot] == MixingType::SemiconvectiveMixing
            {
                k_bot += 1;
            }
            if k_bot < nz
                && model.mixing_type[k_bot] == MixingType::ConvectiveMixing
            {
                let d_avg = 0.5 * (model.d_mix[k] + model.d_mix[k_bot]);
                let vc_avg = 0.5 * (model.conv_vel[k] + model.conv_vel[k_bot]);
                for j in (k + 1)..k_bot {
                    model.mixing_type[j] = MixingType::ConvectiveMixing;
                    model.d_mix[j] = model.d_mix[j].max(d_avg);
                    model.conv_vel[j] = model.conv_vel[j].max(vc_avg);
                    model.mlt_cdc[j] = cdc_factor(model, j) * model.d_mix[j];
                }
            }
            k = k_bot;
        } else {
            k += 1;
        }
    }
}

fn bridge_zone_gaps(
    model: &mut StarModel,
    target: MixingType,
    fill: MixingType,
    max_gap_hp: f64,
) {
    let nz = model.nz;
    let mut k = 1;
    while k + 2 < nz {
        if model.mixing_type[k] == target && model.mixing_type[k + 1] != target
        {
            let mut k_bot = k + 1;
            while k_bot < nz && model.mixing_type[k_bot] != target {
                k_bot += 1;
            }
            if k_bot < nz {
                let dr = (model.r[k] - model.r[k_bot]).abs();
                let hp = model.scale_height[k];
                if dr < max_gap_hp * hp {
                    let d_avg = 0.5 * (model.d_mix[k] + model.d_mix[k_bot]);
                    let vc_avg =
                        0.5 * (model.conv_vel[k] + model.conv_vel[k_bot]);
                    for j in (k + 1)..k_bot {
                        model.d_mix[j] = d_avg;
                        model.conv_vel[j] = vc_avg;
                        model.mlt_cdc[j] = cdc_factor(model, j) * d_avg;
                        model.mixing_type[j] = fill;
                    }
                }
            }
            k = k_bot;
        } else {
            k += 1;
        }
    }
}

fn set_face_radiative(model: &mut StarModel, k: usize) {
    model.d_mix[k] = 0.0;
    model.mlt_d[k] = 0.0;
    model.mlt_cdc[k] = 0.0;
    model.conv_vel[k] = 0.0;
    model.mlt_vc[k] = 0.0;
    model.gradt[k] = model.gradr[k];
    model.mixing_type[k] = MixingType::NoMixing;
}

/// Locates sub-cell convective boundaries (`set_cz_boundary_info` and
/// `locate_convection_boundaries` in `mix_info.f90:424-784`).
///
/// Returns `(cz_bdy_dq, boundaries)` where `cz_bdy_dq` has length `nz` and
/// `boundaries` lists convective boundaries ordered from center to surface.
#[must_use]
pub fn locate_convection_boundaries(
    model: &StarModel,
) -> (Vec<f64>, Vec<ConvBoundary>) {
    let nz = model.nz;
    let mut cz_bdy_dq = vec![0.0; nz];
    if nz < 2 {
        return (cz_bdy_dq, Vec::new());
    }

    // 1. Sub-cell stability zero-crossing offset `cz_bdy_dq[k]` in cell `k`
    // bounded by outer face `k` and inner face `k + 1`.
    for (k, bdy_dq) in cz_bdy_dq.iter_mut().enumerate().take(nz - 1) {
        let mt0 = model.mixing_type[k];
        let mt1 = model.mixing_type[k + 1];
        if mt0 != mt1 {
            let (dg0, dg1) = if mt0 == MixingType::ConvectiveMixing
                || mt1 == MixingType::ConvectiveMixing
            {
                (
                    model.gradl[k] - model.gradr[k],
                    model.gradl[k + 1] - model.gradr[k + 1],
                )
            } else {
                (
                    model.grada[k] - model.gradr[k],
                    model.grada[k + 1] - model.gradr[k + 1],
                )
            };
            if dg0 * dg1 < 0.0 {
                let frac = (-dg0 / (dg1 - dg0)).clamp(0.0, 1.0);
                *bdy_dq = frac * model.dq[k];
            } else {
                *bdy_dq = 0.5 * model.dq[k];
            }
        }
    }

    // 2. Scan outward from center (`nz - 1`) to surface (`0`).
    let mut boundaries = Vec::new();
    let mut in_conv = model.mixing_type[nz - 1] == MixingType::ConvectiveMixing;
    for k in (1..(nz - 1)).rev() {
        let is_conv = model.mixing_type[k] == MixingType::ConvectiveMixing;
        if !in_conv && is_conv {
            // Bottom of a convection zone inside cell `k` (outer face `k` is
            // convective, inner face `k + 1` is non-convective).
            in_conv = true;
            boundaries.push(build_conv_boundary(
                model,
                false,
                k,
                k,
                cz_bdy_dq[k],
            ));
        } else if in_conv && !is_conv {
            // Top of a convection zone inside cell `k` (outer face `k` is
            // non-convective, inner face `k + 1` is convective).
            in_conv = false;
            boundaries.push(build_conv_boundary(
                model,
                true,
                k + 1,
                k,
                cz_bdy_dq[k],
            ));
        }
    }
    if in_conv {
        let is_surf_conv = model.mixing_type[0] == MixingType::ConvectiveMixing;
        let conv_face = usize::from(!is_surf_conv);
        let dq_off = if is_surf_conv { 0.0 } else { cz_bdy_dq[0] };
        boundaries.push(build_conv_boundary(model, true, conv_face, 0, dq_off));
    }

    (cz_bdy_dq, boundaries)
}

fn build_conv_boundary(
    model: &StarModel,
    top_conv_bdy: bool,
    conv_bdy_loc: usize,
    cell_index: usize,
    cz_bdy_dq: f64,
) -> ConvBoundary {
    let nz = model.nz;
    let dq_k = model.dq[cell_index].max(1.0e-99);
    let w = (cz_bdy_dq / dq_k).clamp(0.0, 1.0);
    let r_top = model.r[cell_index];
    let r_bot = if cell_index + 1 < nz {
        model.r[cell_index + 1]
    } else {
        model.r_center
    };
    let r_cb = (w * r_bot.powi(3) + (1.0 - w) * r_top.powi(3))
        .max(0.0)
        .cbrt();
    let conv_bdy_q = (model.q[cell_index] - cz_bdy_dq).clamp(0.0, 1.0);

    // Sub-cell pressure scale height `hp_cb` (`eval_conv_bdy_Hp`).
    let k_next = (cell_index + 1).min(nz - 1);
    let p_cb = (1.0 - w) * model.peos[cell_index] + w * model.peos[k_next];
    let rho_cb = (1.0 - w) * model.rho[cell_index] + w * model.rho[k_next];
    let m_cb = model.m_center + model.xmstar * conv_bdy_q;
    let g_cb = model.cgrav[cell_index] * m_cb / (r_cb * r_cb).max(1.0);
    let hp_cb = if g_cb > 0.0 && rho_cb > 0.0 {
        p_cb / (rho_cb * g_cb)
    } else {
        model.scale_height[cell_index]
    };

    ConvBoundary {
        top_conv_bdy,
        conv_bdy_loc,
        cell_index,
        conv_bdy_q,
        cz_bdy_dq,
        r_cb,
        hp_cb,
    }
}

/// Computes the Lagrangian face mixing coupling factors $\sigma_k$ (`sig`)
/// across all faces `0..nz` (`mix_info.f90:230-355`).
pub fn compute_convection_sigmas(
    model: &mut StarModel,
    dt: f64,
    mix_factor: f64,
    sig_term_limit: f64,
) {
    let nz = model.nz;
    if nz == 0 {
        return;
    }
    model.sig[0] = 0.0;
    model.mlt_cdc[0] = 0.0;
    if mix_factor <= 0.0 {
        model.sig.fill(0.0);
        return;
    }

    let mut max_sig_raw = 0.0_f64;
    for k in 1..nz {
        let dm_bar_k =
            (0.5 * (model.dq[k - 1] + model.dq[k]) * model.xmstar).max(1.0e-99);
        let cdc_k = cdc_factor(model, k) * model.d_mix[k];
        model.mlt_cdc[k] = cdc_k;
        let sig_raw = mix_factor * cdc_k / dm_bar_k;
        model.sig[k] = sig_raw;
        max_sig_raw = max_sig_raw.max(sig_raw);
    }

    if max_sig_raw >= 1.0 && dt > 0.0 && sig_term_limit > 0.0 {
        for k in 1..nz {
            let dq_min = model.dq[k].min(model.dq[k - 1]);
            let sig_lim = sig_term_limit * model.xmstar * dq_min / dt;
            model.sig[k] = model.sig[k].min(sig_lim);
        }
    }
}

/// Evaluates the explicit mass-conserving diffusive species mixing rate
/// $(dX_{j,k}/dt)_{\mathrm{mix}}$ (`set_dxdt_mix` in `mix_info.f90:2303-2364`).
///
/// Returns a flat array of length `nz * species` in zone-major layout.
#[must_use]
pub fn eval_dxdt_mix(model: &StarModel) -> Vec<f64> {
    let nz = model.nz;
    let species = model.species;
    let mut dxdt_mix = vec![0.0; nz * species];
    for k in 0..nz {
        let dm_k = model.dm[k].max(1.0e-99);
        let sig_00 = if k > 0 { model.sig[k] } else { 0.0 };
        let sig_p1 = if k + 1 < nz { model.sig[k + 1] } else { 0.0 };
        for j in 0..species {
            let x_00 = model.xa[k * species + j];
            let flux_in = if k > 0 {
                sig_00 * (model.xa[(k - 1) * species + j] - x_00)
            } else {
                0.0
            };
            let flux_out = if k + 1 < nz {
                sig_p1 * (x_00 - model.xa[(k + 1) * species + j])
            } else {
                0.0
            };
            dxdt_mix[k * species + j] = (flux_in - flux_out) / dm_k;
        }
    }
    dxdt_mix
}
