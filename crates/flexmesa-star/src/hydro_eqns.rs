// SPDX-License-Identifier: GPL-3.0-or-later
//! Coupled 1D Henyey stellar structure and composition residual equations and
//! analytical block-tridiagonal Jacobian assembly (`hydro_eqns.f90`,
//! `hydro_momentum.f90`, `hydro_temperature.f90`, `hydro_energy.f90`,
//! `hydro_chem_eqns.f90`, and `solver_support.f90`).
//!
//! Assembles the dimensionless residual vector `equ` of shape `(nz, nvar)` and
//! the scaled block-tridiagonal Jacobian `(lblk, dblk, ublk)` of shape
//! `(nz, nvar, nvar)` using 3-cell stencil forward-mode automatic
//! differentiation ([`StarAdOrder1`]).

use flexmesa_const::PI;

use crate::controls::{StarError, VarLayout};
use crate::eps_grav::eval_eps_grav;
pub use crate::hydro_energy::do1_energy_eqn;
pub use crate::hydro_momentum::{
    do1_momentum_eqn, do1_radius_eqn, eval_surf_momentum_bc_pressure_ad,
    has_surf_momentum_bc,
};
pub use crate::hydro_temperature::do1_temperature_eqn;
use crate::hydro_vars::set_hydro_vars;
use crate::model::StarModel;
use crate::star_ad::{
    I_HP_00, I_HP_M1, I_HP_P1, I_L_00, I_L_M1, I_L_P1, I_LND_00, I_LND_M1,
    I_LND_P1, I_LNR_00, I_LNR_M1, I_LNR_P1, I_LNT_00, I_LNT_M1, I_LNT_P1,
    I_V_00, I_V_M1, I_V_P1, I_W_00, I_W_M1, I_W_P1, StarAdExt, StarAdOrder1,
};

/// Assembled 1D Henyey residual vector and scaled block-tridiagonal Jacobian
/// $L_k \delta\tilde{x}_{k-1} + D_k \delta\tilde{x}_k
/// + U_k \delta\tilde{x}_{k+1} = -\mathrm{equ}_k$.
#[derive(Debug, Clone)]
pub struct HenyeySystem {
    /// Number of coupled variables per zone (`nvar_hydro` or `nvar_total`).
    pub nvar: usize,
    /// Number of Lagrangian zones (`nz`).
    pub nz: usize,
    /// Variable scaling factors `x_scale[k * nvar + j]` (`solver_support.f90:312`).
    pub x_scale: Vec<f64>,
    /// Residual vector `equ[k * nvar + i]` (`len = nz * nvar`).
    pub equ: Vec<f64>,
    /// Sub-diagonal Jacobian blocks `lblk[k * nvar * nvar + i * nvar + j]`.
    pub lblk: Vec<f64>,
    /// Diagonal Jacobian blocks `dblk[k * nvar * nvar + i * nvar + j]`.
    pub dblk: Vec<f64>,
    /// Super-diagonal Jacobian blocks `ublk[k * nvar * nvar + i * nvar + j]`.
    pub ublk: Vec<f64>,
}

impl HenyeySystem {
    /// Allocates a zero-initialized [`HenyeySystem`] for `(nz, nvar)` with
    /// `x_scale`.
    #[must_use]
    pub fn new(nz: usize, nvar: usize, x_scale: Vec<f64>) -> Self {
        let blk_len = nz * nvar * nvar;
        Self {
            nvar,
            nz,
            x_scale,
            equ: vec![0.0; nz * nvar],
            lblk: vec![0.0; blk_len],
            dblk: vec![0.0; blk_len],
            ublk: vec![0.0; blk_len],
        }
    }

    /// Accumulates a scalar partial derivative
    /// $\partial \mathrm{equ}_{k,\mathrm{equ}} / \partial x_{k_{\mathrm{var}}, j_{\mathrm{var}}}$
    /// scaled by `x_scale[k_var, j_var]` into `lblk`, `dblk`, or `ublk`.
    #[inline]
    pub fn add_partial(
        &mut self,
        k_equ: usize,
        i_equ: usize,
        k_var: usize,
        j_var: usize,
        deriv: f64,
    ) {
        // `e00`/`em1`/`ep1` drop partials below `1d-250`
        // (`star/private/star_utils.f90:3056, 3106, 3155`).
        if deriv.abs() < 1.0e-250 {
            return;
        }
        let nvar = self.nvar;
        let blk_sz = nvar * nvar;
        let scaled = deriv * self.x_scale[k_var * nvar + j_var];
        let idx = k_equ * blk_sz + i_equ * nvar + j_var;
        if k_var == k_equ {
            self.dblk[idx] += scaled;
        } else if k_var + 1 == k_equ {
            self.lblk[idx] += scaled;
        } else if k_equ + 1 == k_var {
            self.ublk[idx] += scaled;
        }
    }

    /// Accumulates 3-cell stencil partials `(d_m1, d_00, d_p1)` for variable
    /// slot `j_var` at equation `(k, i_equ)`.
    #[inline]
    fn add_stencil_partials(
        &mut self,
        k: usize,
        i_equ: usize,
        j_var: usize,
        d_m1: f64,
        d_00: f64,
        d_p1: f64,
    ) {
        self.add_partial(k, i_equ, k, j_var, d_00);
        if k > 0 {
            self.add_partial(k, i_equ, k - 1, j_var, d_m1);
        }
        if k + 1 < self.nz {
            self.add_partial(k, i_equ, k + 1, j_var, d_p1);
        }
    }

    /// Stores `equ[k, i_equ] = resid.val` and unpacks the 3-cell stencil
    /// [`StarAdOrder1`] partials into `lblk`, `dblk`, and `ublk`
    /// (`save_eqn_residual_info` in `star/private/star_utils.f90:1603-1614`).
    pub fn save_eqn_residual(
        &mut self,
        layout: &VarLayout,
        k: usize,
        i_equ: usize,
        resid: &StarAdOrder1,
    ) {
        self.equ[k * self.nvar + i_equ] = resid.val;
        let slots = [
            (Some(layout.i_lnd), I_LND_M1, I_LND_00, I_LND_P1),
            (Some(layout.i_lnt), I_LNT_M1, I_LNT_00, I_LNT_P1),
            (Some(layout.i_lnr), I_LNR_M1, I_LNR_00, I_LNR_P1),
            (Some(layout.i_lum), I_L_M1, I_L_00, I_L_P1),
            (layout.i_v.or(layout.i_u), I_V_M1, I_V_00, I_V_P1),
            (layout.i_w, I_W_M1, I_W_00, I_W_P1),
            (layout.i_hp, I_HP_M1, I_HP_00, I_HP_P1),
        ];
        for (j_opt, m1_slot, d0_slot, p1_slot) in slots {
            if let Some(j_var) = j_opt {
                self.add_stencil_partials(
                    k,
                    i_equ,
                    j_var,
                    resid.d1[m1_slot],
                    resid.d1[d0_slot],
                    resid.d1[p1_slot],
                );
            }
        }
    }
    /// Copies `other` into `self`, reusing allocated capacities across Newton
    /// backtracking evaluations.
    pub fn copy_from(&mut self, other: &Self) {
        self.nvar = other.nvar;
        self.nz = other.nz;
        self.x_scale.clone_from(&other.x_scale);
        self.equ.clone_from(&other.equ);
        self.lblk.clone_from(&other.lblk);
        self.dblk.clone_from(&other.dblk);
        self.ublk.clone_from(&other.ublk);
    }
}

/// Populates the per-zone variable scaling vector `x_scale` in place
/// (`set_xscale_info` in `star/private/solver_support.f90:42-86`).
pub fn compute_x_scale_into(
    model: &StarModel,
    include_chem: bool,
    x_scale: &mut Vec<f64>,
) {
    let nz = model.nz;
    let layout = &model.layout;
    let nvar = layout.nvar(include_chem);
    x_scale.clear();
    x_scale.resize(nz * nvar, 1.0);

    for k in 0..nz {
        let base = k * nvar;
        let lnd_s = model
            .lnd_start
            .get(k)
            .copied()
            .unwrap_or(model.lnd[k])
            .abs();
        let lnt_s = model
            .ln_t_start
            .get(k)
            .copied()
            .unwrap_or(model.ln_t[k])
            .abs();
        let lnr_s = model
            .ln_r_start
            .get(k)
            .copied()
            .unwrap_or(model.ln_r[k])
            .abs();
        let lum_s = model
            .lum_start
            .get(k)
            .copied()
            .unwrap_or(model.lum[k])
            .abs();
        x_scale[base + layout.i_lnd] = lnd_s.max(1.0);
        x_scale[base + layout.i_lnt] = lnt_s.max(1.0);
        x_scale[base + layout.i_lnr] = lnr_s.max(1.0);
        x_scale[base + layout.i_lum] = lum_s.max(1.0);
        if let Some(iv) = layout.i_v {
            let v_s = model
                .v_start
                .get(k)
                .copied()
                .unwrap_or_else(|| model.v.get(k).copied().unwrap_or(0.0))
                .abs();
            x_scale[base + iv] = v_s.max(1.0);
        }
        if let Some(iu) = layout.i_u {
            let u_s = model
                .u_start
                .get(k)
                .copied()
                .unwrap_or_else(|| model.u.get(k).copied().unwrap_or(0.0))
                .abs();
            x_scale[base + iu] = u_s.max(1.0);
        }
        if let Some(iw) = layout.i_w {
            let w_s = model
                .aux
                .rsp
                .w_start
                .get(k)
                .copied()
                .unwrap_or_else(|| {
                    model
                        .aux
                        .rsp
                        .et_rsp
                        .get(k)
                        .copied()
                        .unwrap_or(0.0)
                        .max(0.0)
                        .sqrt()
                })
                .abs();
            x_scale[base + iw] = w_s.max(1.0);
        }
        if let Some(ihp) = layout.i_hp {
            let hp = model
                .xh_start
                .get(k * layout.nvar_hydro + ihp)
                .copied()
                .unwrap_or_else(|| {
                    model.aux.rsp.hp_face.get(k).copied().unwrap_or(1.0)
                })
                .abs();
            x_scale[base + ihp] = hp.max(1.0);
        }
        if let Some(ia) = layout.i_alpha_rti {
            let a_s = model
                .aux
                .alpha_rti_start
                .get(k)
                .copied()
                .unwrap_or_else(|| {
                    model.aux.alpha_rti.get(k).copied().unwrap_or(0.0)
                })
                .abs();
            x_scale[base + ia] = a_s.max(1.0);
        }
        if include_chem {
            let xa_min = model.controls.xa_scale;
            for j in 0..model.species {
                let ivar = layout.chem_var_index(j);
                let xa_s = model
                    .xa_start
                    .get(k * model.species + j)
                    .copied()
                    .unwrap_or_else(|| model.xa[k * model.species + j]);
                x_scale[base + ivar] = xa_s.max(xa_min);
            }
        }
    }
}

/// Computes the per-zone variable scaling vector `x_scale` (`set_xscale_info`
/// in `star/private/solver_support.f90:312-363`).
#[must_use]
pub fn compute_x_scale(model: &StarModel, include_chem: bool) -> Vec<f64> {
    let mut x_scale = Vec::new();
    compute_x_scale_into(model, include_chem, &mut x_scale);
    x_scale
}

/// Continuity / volume equation for cell `k` (`do1_density_eqn` in
/// `star/private/hydro_eqns.f90:550-641`):
/// $$\mathrm{equ}(\mathrm{i\_dlnd\_dt}, k) =
/// \ln R_k - \frac{1}{3}\ln\!\left(r_{k+1}^3 + \frac{3\,\Delta m_k}{4\pi \rho_k}\right).$$
///
/// # Errors
/// Returns [`StarError::ConvergenceFailure`] if the trial volume is
/// non-positive.
pub fn do1_density_eqn(
    model: &StarModel,
    k: usize,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    // `do1_density_eqn` (`star/private/hydro_eqns.f90:566-583`).
    let lnr_00 = StarAdOrder1::variable(model.ln_r[k], I_LNR_00);
    let rho = crate::hydro_momentum::wrap_d_00(model, k);
    let dr3 = (model.dm[k] / rho) / (4.0 * PI / 3.0);
    // `wrap_r_p1`: `r(k+1)`, or `R_center` below the innermost cell.
    let r_p1 = if k + 1 < model.nz {
        let r = model.r[k + 1];
        let mut out = StarAdOrder1::constant(r);
        out.d1[I_LNR_P1] = r;
        out
    } else {
        StarAdOrder1::constant(model.r_center)
    };
    let rp13 = r_p1.pow3();
    let rp13_p_dr3 = rp13 + dr3;
    if rp13_p_dr3.val <= 0.0 || !rp13_p_dr3.val.is_finite() {
        return Err(StarError::ConvergenceFailure(format!(
            "non-positive inner volume at zone {k}: {}",
            rp13_p_dr3.val
        )));
    }
    let resid = lnr_00 - rp13_p_dr3.ln() / 3.0;
    sys.save_eqn_residual(&model.layout, k, model.layout.i_dlnd_dt, &resid);
    Ok(())
}

/// Outer-boundary pressure and temperature with partials, as `get_PT_bc_ad`
/// builds them (`star/private/hydro_eqns.f90:817-999`).
#[derive(Debug, Clone, Copy)]
pub struct SurfBcAd {
    /// `P_bc_ad`.
    pub p_bc: StarAdOrder1,
    /// `lnP_bc_ad`.
    pub ln_p_bc: StarAdOrder1,
    /// `T_bc_ad`.
    pub t_bc: StarAdOrder1,
    /// `lnT_bc_ad`.
    pub ln_t_bc: StarAdOrder1,
}

/// `get_PT_bc_ad` (`star/private/hydro_eqns.f90:817-999`): offsets
/// `P_surf` and `T_surf` ([`StarModel::surf_pt`], from `get_surf_PT`) to the
/// centre of cell 1 and chains the atmosphere partials into the cell-1
/// variables in upstream's order of operations.
#[must_use]
pub fn get_pt_bc_ad(model: &StarModel) -> SurfBcAd {
    // `star/private/hydro_eqns.f90:782-786`.
    let offset_p_to_cell_center = !model.controls.use_momentum_outer_bc;
    let offset_t_to_cell_center = !(model.controls.use_other_surface_pt
        || model.controls.rsp2_flag
        || model.controls.use_rsp_l_eqn_outer_bc);
    let sp = model.surf_pt;
    let r = model.r[0];

    // `hydro_eqns.f90:851-868`.
    let p_surf = flexmesa_math::exp(sp.ln_p);
    let t_surf = flexmesa_math::exp(sp.ln_t);
    let mut dp0 = 0.0;
    let mut dt0 = 0.0;
    if offset_p_to_cell_center {
        dp0 = model.cgrav[0] * model.m_grav[0] * model.dm[0]
            / (8.0 * PI * flexmesa_math::pow4(r));
    }
    let (t_1, peos_1, gradt_1) = (model.t[0], model.peos[0], model.gradt[0]);
    if offset_t_to_cell_center {
        dt0 = dp0 * gradt_1 * t_1 / peos_1;
    }
    let p_bc = p_surf + dp0;
    let t_bc = t_surf + dt0;
    let ln_p_bc = flexmesa_math::log(p_bc);
    let ln_t_bc = flexmesa_math::log(t_bc);

    // `hydro_eqns.f90:889-914`.
    let mut dp0_dlnr = 0.0;
    if offset_p_to_cell_center {
        dp0_dlnr = -4.0 * dp0;
    }
    let mut dt0_dlnr = 0.0;
    let mut dt0_dlnt = 0.0;
    let mut dt0_dlnd = 0.0;
    let mut dt0_dl = 0.0;
    if offset_t_to_cell_center {
        let g = &model.gradt_ad[0].d1;
        // `chiT_for_partials(1)` and `chiRho_for_partials(1)` are the
        // `lnT_00` and `lnd_00` partials of `ln_peos_ad`.
        let lp = &model.ln_peos_ad[0].d1;
        let d_gradt_dlnr = g[I_LNR_00];
        let d_gradt_dlnt00 = g[I_LNT_00];
        let d_gradt_dlnd00 = g[I_LND_00];
        let d_gradt_dl = g[I_L_00];
        dt0_dlnr = -4.0 * dt0 + dp0 * d_gradt_dlnr * t_1 / peos_1;
        let dpinv_dlnt = -lp[I_LNT_00] / peos_1;
        dt0_dlnt = dt0
            + dp0 * d_gradt_dlnt00 * t_1 / peos_1
            + dp0 * gradt_1 * t_1 * dpinv_dlnt;
        let dpinv_dlnd = -lp[I_LND_00] / peos_1;
        dt0_dlnd = dp0 * d_gradt_dlnd00 * t_1 / peos_1
            + dp0 * gradt_1 * t_1 * dpinv_dlnd;
        dt0_dl = dp0 * d_gradt_dl * t_1 / peos_1;
    }

    // `hydro_eqns.f90:916-923`.
    let dlnp_bc_dp0 = 1.0 / p_bc;
    let dlnt_bc_dt0 = 1.0 / t_bc;
    let dlnp_bc_dlnpsurf = p_surf / p_bc;
    let dlnt_bc_dlntsurf = t_surf / t_bc;
    let dlnkap_dlnd = model.d_opacity_dlnd[0] / model.opacity[0];
    let dlnkap_dlnt = model.d_opacity_dlnt[0] / model.opacity[0];

    // `hydro_eqns.f90:925-928`.
    let dlnp_bc_dlnd = dlnp_bc_dlnpsurf * sp.dlnp_dlnkap * dlnkap_dlnd;
    let dlnp_bc_dlnt = dlnp_bc_dlnpsurf * sp.dlnp_dlnkap * dlnkap_dlnt;
    let dlnp_bc_dl = dlnp_bc_dlnpsurf * sp.dlnp_dl;
    let dlnp_bc_dlnr = dlnp_bc_dlnpsurf * sp.dlnp_dlnr + dlnp_bc_dp0 * dp0_dlnr;

    // `hydro_eqns.f90:930-959`.
    let p_bc_ad = wrap_surf_bc(
        p_bc,
        p_bc * dlnp_bc_dlnd,
        p_bc * dlnp_bc_dlnt,
        p_bc * dlnp_bc_dlnr,
        p_bc * dlnp_bc_dl,
    );
    let ln_p_bc_ad = wrap_surf_bc(
        ln_p_bc,
        dlnp_bc_dlnd,
        dlnp_bc_dlnt,
        dlnp_bc_dlnr,
        dlnp_bc_dl,
    );

    // `hydro_eqns.f90:961-966`.
    let dlnt_bc_dlnt = dlnt_bc_dlntsurf * sp.dlnt_dlnkap * dlnkap_dlnt
        + dlnt_bc_dt0 * dt0_dlnt;
    let dlnt_bc_dlnd = dlnt_bc_dlntsurf * sp.dlnt_dlnkap * dlnkap_dlnd
        + dlnt_bc_dt0 * dt0_dlnd;
    let dlnt_bc_dl = dlnt_bc_dlntsurf * sp.dlnt_dl + dlnt_bc_dt0 * dt0_dl;
    let dlnt_bc_dlnr = dlnt_bc_dlntsurf * sp.dlnt_dlnr + dlnt_bc_dt0 * dt0_dlnr;

    // `hydro_eqns.f90:968-997`.
    let t_bc_ad = wrap_surf_bc(
        t_bc,
        t_bc * dlnt_bc_dlnd,
        t_bc * dlnt_bc_dlnt,
        t_bc * dlnt_bc_dlnr,
        t_bc * dlnt_bc_dl,
    );
    let ln_t_bc_ad = wrap_surf_bc(
        ln_t_bc,
        dlnt_bc_dlnd,
        dlnt_bc_dlnt,
        dlnt_bc_dlnr,
        dlnt_bc_dl,
    );
    SurfBcAd {
        p_bc: p_bc_ad,
        ln_p_bc: ln_p_bc_ad,
        t_bc: t_bc_ad,
        ln_t_bc: ln_t_bc_ad,
    }
}

/// `wrap(x, val, ...)` with only the `lnd_00`, `lnT_00`, `lnR_00` and `L_00`
/// partials set, as `get_PT_bc_ad` calls it.
fn wrap_surf_bc(
    val: f64,
    dlnd00: f64,
    dlnt00: f64,
    dlnr00: f64,
    dl00: f64,
) -> StarAdOrder1 {
    let mut out = StarAdOrder1::constant(val);
    out.d1[I_LND_00] = dlnd00;
    out.d1[I_LNT_00] = dlnt00;
    out.d1[I_LNR_00] = dlnr00;
    out.d1[I_L_00] = dl00;
    out
}

/// Surface pressure and temperature boundary condition equations at `k = 0`
/// (`set_Psurf_BC` and `set_Tsurf_BC` in `star/private/hydro_eqns.f90:1002-1067`).
pub fn do_surf_bc_eqns(model: &StarModel, sys: &mut HenyeySystem) {
    if model.nz == 0 {
        return;
    }
    let bc = get_pt_bc_ad(model);

    if model.controls.use_compression_outer_bc && model.nz >= 2 {
        let lnd_0 = StarAdOrder1::variable(model.lnd[0], I_LND_00);
        let lnd_1 = StarAdOrder1::variable(model.lnd[1], I_LND_P1);
        let rho_0 = lnd_0.exp();
        let rho_1 = lnd_1.exp();
        let dlnd_0 =
            lnd_0 - model.lnd_start.first().copied().unwrap_or(model.lnd[0]);
        let dlnd_1 =
            lnd_1 - model.lnd_start.get(1).copied().unwrap_or(model.lnd[1]);
        let resid_c = (rho_1 * dlnd_0 - rho_0 * dlnd_1) / model.dt.max(1.0e-30);
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_dv_dt, &resid_c);
    } else if model.controls.use_fixed_vsurf_outer_bc
        && (model.controls.v_flag || model.controls.u_flag)
    {
        let vsurf_val = if model.controls.v_flag {
            model.v[0]
        } else {
            model.u[0]
        };
        let vsurf = StarAdOrder1::variable(vsurf_val, I_V_00);
        let cs_0 = model
            .aux
            .csound_start
            .first()
            .copied()
            .filter(|&c| c > 0.0)
            .unwrap_or(model.csound[0])
            .max(1.0);
        let resid_v = (vsurf - model.controls.fixed_vsurf) / cs_0;
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_dv_dt, &resid_v);
    } else if !has_surf_momentum_bc(&model.controls) {
        let ln_p_bc = bc.ln_p_bc;
        let ln_peos_0 = model.ln_peos_ad[0];
        let denom_p = if ln_peos_0.val > 1.0 {
            ln_peos_0
        } else {
            StarAdOrder1::ONE
        };
        let resid_p = (ln_p_bc - ln_peos_0) / denom_p;
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_dv_dt, &resid_p);
    }

    let skip_surf_t_bc = model.controls.use_rsp_l_eqn_outer_bc
        || (model.controls.rsp2_flag
            && model.controls.rsp2.use_l_eqn_at_surface);
    if !skip_surf_t_bc {
        let lnt_0 = StarAdOrder1::variable(model.ln_t[0], I_LNT_00);
        let ln_t_bc = bc.ln_t_bc;
        let denom_t = if lnt_0.val > 1.0 {
            lnt_0
        } else {
            StarAdOrder1::ONE
        };
        let resid_t = (ln_t_bc - lnt_0) / denom_t;
        sys.save_eqn_residual(&model.layout, 0, model.layout.i_equ_l, &resid_t);
    }
}

/// Coupled nuclear burning and diffusive mixing equations for cell `k`
/// (`do1_chem_eqns` in `star/private/hydro_chem_eqns.f90:50-222`).
///
/// Combines start-of-step Lagrangian diffusive mixing $\sigma_k$ across faces
/// `k` and `k + 1` (gated by `do_mix` and `mix_factor`) with the
/// reaction-network abundance rates $\dot{X}_{\mathrm{nuc},j,k}$ and their
/// partial derivatives with respect to $(\ln\rho_k, \ln T_k, X_{m,k})$ (gated
/// by `do_burn`, `dxdt_nuc_factor`, and `!doing_op_split_burn`).
pub fn do1_chem_eqns(model: &StarModel, k: usize, sys: &mut HenyeySystem) {
    let nz = model.nz;
    let species = model.species;
    let dt = model.dt.max(1.0e-30);
    let dm_k = model.dm[k].max(1.0e-30);

    let t_start_k = flexmesa_math::exp(
        model.ln_t_start.get(k).copied().unwrap_or(model.ln_t[k]),
    );
    let doing_op_split_burn = model.controls.op_split_burn
        && t_start_k >= model.controls.op_split_burn_min_t;
    let active_burn = model.controls.do_burn
        && model.controls.dxdt_nuc_factor != 0.0
        && !doing_op_split_burn;

    let sig_00 = if model.controls.do_mix && k > 0 {
        model.sig[k]
    } else {
        0.0
    };
    let sig_p1 = if model.controls.do_mix && k + 1 < nz {
        model.sig[k + 1]
    } else {
        0.0
    };
    // `set_dxdt_mix` (`star/private/mix_info.f90:2318-2345`).
    let dflux00_dxm1 = if k > 0 { -sig_00 } else { 0.0 };
    let dflux00_dx00 = if k > 0 { sig_00 } else { 0.0 };
    let (dfluxp1_dx00, dfluxp1_dxp1) =
        if k + 1 < nz { (-sig_p1, sig_p1) } else { (0.0, 0.0) };
    let d_dxdt_mix_dx00 = (dfluxp1_dx00 - dflux00_dx00) / dm_k;
    let d_dxdt_mix_dxm1 = -dflux00_dxm1 / dm_k;
    let d_dxdt_mix_dxp1 = dfluxp1_dxp1 / dm_k;

    let has_dx = model.xa_sub_xa_start.len() == model.xa.len()
        && model.xa_start.len() == model.xa.len();

    for j in 0..species {
        let kj = k * species + j;
        let i_var = model.layout.chem_var_index(j);
        let i_equ = i_var;
        let nuc_base = k * species * species + j * species;
        let x_scale_kj = sys.x_scale[k * sys.nvar + i_var];
        let eqn_scale = model
            .controls
            .min_chem_eqn_scale
            .max(x_scale_kj / dt)
            .max(1.0e-30);

        let x_00 = model.xa[kj];
        let f_00 = if k > 0 {
            sig_00 * (model.xa[(k - 1) * species + j] - x_00)
        } else {
            0.0
        };
        let f_p1 = if k + 1 < nz {
            sig_p1 * (x_00 - model.xa[(k + 1) * species + j])
        } else {
            0.0
        };
        let dxdt_mix = (f_00 - f_p1) / dm_k;
        let dxdt_nuc = if active_burn { model.dxdt_nuc[kj] } else { 0.0 };
        let dxdt_actual = if has_dx {
            model.xa_sub_xa_start[kj]
        } else {
            model.xa[kj] - model.xa_start[kj]
        } / dt;
        sys.equ[k * sys.nvar + i_equ] =
            (dxdt_mix + dxdt_nuc - dxdt_actual) / eqn_scale;

        // Partials in upstream order (`star/private/hydro_chem_eqns.f90`,
        // `do1_chem_eqns`): `e00(i, i, -1d0/eqn_scale/dt)`, then the burning
        // partials, then the mixing partials.
        sys.add_partial(k, i_equ, k, i_var, -1.0 / eqn_scale / dt);

        if active_burn {
            for m in 0..species {
                let d_nuc = model.d_dxdt_nuc_dx[nuc_base + m];
                sys.add_partial(
                    k,
                    i_equ,
                    k,
                    model.layout.chem_var_index(m),
                    d_nuc / eqn_scale,
                );
            }
            sys.add_partial(
                k,
                i_equ,
                k,
                model.layout.i_lnd,
                model.d_dxdt_nuc_dlnd[kj] / eqn_scale,
            );
            sys.add_partial(
                k,
                i_equ,
                k,
                model.layout.i_lnt,
                model.d_dxdt_nuc_dlnt[kj] / eqn_scale,
            );
        }

        if model.controls.do_mix {
            sys.add_partial(k, i_equ, k, i_var, d_dxdt_mix_dx00 / eqn_scale);
            if k > 0 {
                sys.add_partial(k, i_equ, k - 1, i_var, d_dxdt_mix_dxm1 / eqn_scale);
            }
            if k + 1 < nz {
                sys.add_partial(k, i_equ, k + 1, i_var, d_dxdt_mix_dxp1 / eqn_scale);
            }
        }
    }
}

/// Assembles the coupled 1D Henyey residual vector and scaled block-tridiagonal
/// Jacobian blocks into `sys` from the current microphysical and [`StarAdOrder1`]
/// fields on `model` without re-running `set_hydro_vars` or `eval_eps_grav`.
///
/// # Errors
/// Returns [`StarError`] if the continuity volume equation encounters a
/// non-positive trial shell volume.
pub fn assemble_henyey_system(
    model: &mut StarModel,
    include_chem: bool,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    let nz = model.nz;
    let nvar = model.layout.nvar(include_chem);
    sys.nvar = nvar;
    sys.nz = nz;
    compute_x_scale_into(model, include_chem, &mut sys.x_scale);
    sys.equ.clear();
    sys.equ.resize(nz * nvar, 0.0);
    let blk_len = nz * nvar * nvar;
    sys.lblk.clear();
    sys.lblk.resize(blk_len, 0.0);
    sys.dblk.clear();
    sys.dblk.resize(blk_len, 0.0);
    sys.ublk.clear();
    sys.ublk.resize(blk_len, 0.0);

    do_surf_bc_eqns(model, sys);
    let riemann_p_surf = (model.controls.u_flag
        && !model.controls.v_flag
        && !model.controls.use_compression_outer_bc
        && !model.controls.use_fixed_vsurf_outer_bc
        && has_surf_momentum_bc(&model.controls))
    .then(|| eval_surf_momentum_bc_pressure_ad(model));

    let mut tot_e_equ_err = 0.0_f64;
    for k in 0..nz {
        do1_density_eqn(model, k, sys)?;
        if model.controls.u_flag && !model.controls.v_flag {
            crate::riemann_ad::do1_riemann_momentum_eqn(
                model,
                k,
                if k == 0 { riemann_p_surf } else { None },
                sys,
            );
        } else {
            do1_momentum_eqn(model, k, sys);
        }
        if !model.controls.rsp2_flag {
            do1_temperature_eqn(model, k, sys);
        }
        tot_e_equ_err += crate::hydro_energy::do1_energy_eqn_with_ergs_error(
            model,
            k,
            include_chem,
            sys,
        );
        do1_radius_eqn(model, k, sys);
        if model.controls.rti_flag {
            crate::hydro_rti::do1_dalpha_rti_dt_eqn(model, k, sys);
        }
        if include_chem {
            do1_chem_eqns(model, k, sys);
        }
    }
    model.aux.step_hist.tot_e_equ_err = tot_e_equ_err;
    if model.controls.rsp2_flag {
        crate::rsp2_ad::eval_and_save_rsp2_eqns(model, sys);
    }
    Ok(())
}

/// Evaluates microphysics (`set_hydro_vars`), gravitational energy generation
/// (`eval_eps_grav`), and all coupled 1D Henyey residual equations and scaled
/// block-tridiagonal Jacobian blocks in place into `sys` (`eval_equations` in
/// `star/private/hydro_eqns.f90:91-265`).
///
/// # Errors
/// Returns [`StarError`] if microphysics or volume equation evaluation fails.
pub fn eval_henyey_equations_into(
    model: &mut StarModel,
    include_chem: bool,
    sys: &mut HenyeySystem,
) -> Result<(), StarError> {
    // `eval_equations` skips `set_solver_vars` before the first iteration
    // (`star/private/solver_support.f90:108-118`); every later evaluation
    // sets the variables, so the flag is restored here.
    if model.aux.need_to_setvars {
        set_hydro_vars(model, false, !include_chem)?;
    }
    model.aux.need_to_setvars = true;
    // `star/private/hydro_eqns.f90:131-137`: revise EOS composition partials
    // for PC/Skye/ideal zones before assembling energy and momentum equations.
    crate::micro::fix_d_eos_dxa_partials(model)?;
    // `star/private/hydro_eqns.f90:227-230` and
    // `star/private/hydro_energy.f90:394-416`: evaluate `eps_grav` only when
    // the energy equation uses the `eps_grav` form; zero it otherwise.
    let uses_eps_grav = crate::hydro_energy::eps_grav_form(model);
    if uses_eps_grav {
        eval_eps_grav(model)?;
    } else {
        model.eps_grav_ad.fill(StarAdOrder1::ZERO);
        model.aux.d_eps_grav_dx.fill(0.0);
    }
    assemble_henyey_system(model, include_chem, sys)
}

/// Evaluates microphysics (`set_hydro_vars`), gravitational energy generation
/// (`eval_eps_grav`), and returns a newly allocated [`HenyeySystem`].
///
/// # Errors
/// Returns [`StarError`] if microphysics or volume equation evaluation fails.
pub fn eval_henyey_equations(
    model: &mut StarModel,
    include_chem: bool,
) -> Result<HenyeySystem, StarError> {
    let mut sys = HenyeySystem::new(0, 0, Vec::new());
    eval_henyey_equations_into(model, include_chem, &mut sys)?;
    Ok(sys)
}
