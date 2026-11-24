// SPDX-License-Identifier: GPL-3.0-or-later
//! Rotationally induced mixing coefficients (Heger, Langer & Woosley 2000;
//! Spruit 2002): `set_rotation_mixing_info` and its helpers in
//! `star/private/rotation_mix_info.f90:47-1369`, with `weighted_smoothing`
//! (`star/private/star_utils.f90:2777-2830`).
//!
//! Every array is face-centred, indexed `0..nz` from the surface inwards
//! (upstream `1..nz`). Diffusion coefficients are in cm^2 s^-1.

use flexmesa_const::{
    MixingType, CLIGHT, CRAD, L_SUN, M_SUN, ONE_THIRD, PI, PI4, R_SUN,
};

use crate::controls::StarError;
use crate::model::StarModel;

/// Critical Richardson number (`Ri_crit`, `rotation_mix_info.f90:35`).
const RI_CRIT: f64 = 0.25;
/// Critical Reynolds number (`R_crit`, `rotation_mix_info.f90:36`).
const R_CRIT: f64 = 2500.0;

/// Outputs of `set_rotation_mixing_info` kept on the star
/// (`s% D_visc`, `s% D_DSI`, ..., `s% omega_shear`, `s% richardson_number`,
/// `s% domega_dlnR`). They are recomputed by every `set_mixing_info`, so they
/// are not part of a photo.
#[derive(Debug, Clone, Default)]
pub struct RotMixArrays {
    /// Kinematic shear viscosity (`s% D_visc`).
    pub d_visc: Vec<f64>,
    /// Dynamical shear instability (`s% D_DSI`).
    pub d_dsi: Vec<f64>,
    /// Solberg-Hoiland instability (`s% D_SH`).
    pub d_sh: Vec<f64>,
    /// Secular shear instability (`s% D_SSI`).
    pub d_ssi: Vec<f64>,
    /// Eddington-Sweet circulation (`s% D_ES`).
    pub d_es: Vec<f64>,
    /// Goldreich-Schubert-Fricke instability (`s% D_GSF`).
    pub d_gsf: Vec<f64>,
    /// Spruit-Tayler dynamo diffusivity (`s% D_ST`).
    pub d_st: Vec<f64>,
    /// Spruit-Tayler dynamo viscosity (`s% nu_ST`).
    pub nu_st: Vec<f64>,
    /// Dynamo radial field (`s% dynamo_B_r`, gauss).
    pub dynamo_b_r: Vec<f64>,
    /// Dynamo azimuthal field (`s% dynamo_B_phi`, gauss).
    pub dynamo_b_phi: Vec<f64>,
    /// `|dlnOmega/dlnr|` clipped to `[1e-30, 1e30]` (`s% omega_shear`).
    pub omega_shear: Vec<f64>,
    /// Richardson number (`s% richardson_number`).
    pub richardson_number: Vec<f64>,
    /// `dOmega/dlnR` (`s% domega_dlnR`).
    pub domega_dlnr: Vec<f64>,
    /// `dlnRho/dlnP` (`s% grad_density`, set by `set_grads`).
    pub grad_density: Vec<f64>,
    /// `dlnT/dlnP` (`s% grad_temperature`, set by `set_grads`).
    pub grad_temperature: Vec<f64>,
    /// Rotational part of `D_mix` (`s% D_mix_rotation`).
    pub d_mix_rotation: Vec<f64>,
}

impl RotMixArrays {
    fn resize(&mut self, nz: usize) {
        for v in [
            &mut self.d_visc,
            &mut self.d_dsi,
            &mut self.d_sh,
            &mut self.d_ssi,
            &mut self.d_es,
            &mut self.d_gsf,
            &mut self.d_st,
            &mut self.nu_st,
            &mut self.dynamo_b_r,
            &mut self.dynamo_b_phi,
            &mut self.omega_shear,
            &mut self.richardson_number,
            &mut self.domega_dlnr,
            &mut self.grad_density,
            &mut self.grad_temperature,
            &mut self.d_mix_rotation,
        ] {
            v.resize(nz, 0.0);
        }
    }
}

/// Spruit-Tayler state carried between steps for its time smoothing:
/// `D_ST_start`, `nu_ST_start` and `have_ST_start_info` set after a converged
/// `do_struct_burn_mix` (`star/private/struct_burn_mix.f90:192-198`), and the
/// `prev_mesh_*` copies `prepare_for_new_step` takes before remeshing
/// (`star/private/evolve.f90:1859-1879`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StStartInfo {
    /// `s% D_ST_start`.
    pub d_st_start: Vec<f64>,
    /// `s% nu_ST_start`.
    pub nu_st_start: Vec<f64>,
    /// `s% have_ST_start_info`.
    pub have_st_start_info: bool,
    /// `s% prev_mesh_D_ST_start`.
    pub prev_mesh_d_st_start: Vec<f64>,
    /// `s% prev_mesh_nu_ST_start`.
    pub prev_mesh_nu_st_start: Vec<f64>,
    /// `s% prev_mesh_dq`.
    pub prev_mesh_dq: Vec<f64>,
    /// `s% prev_mesh_nz`.
    pub prev_mesh_nz: usize,
    /// `s% prev_mesh_have_ST_start_info`.
    pub prev_mesh_have_st_start_info: bool,
}

/// Stores the ST mixing info after a converged `do_struct_burn_mix`
/// (`star/private/struct_burn_mix.f90:192-198`).
pub fn store_st_start_info(model: &mut StarModel) {
    if !model.controls.rotation_flag {
        return;
    }
    let nz = model.nz;
    let rm = &model.aux.rotmix;
    let st = &mut model.aux.st_start;
    st.d_st_start.clear();
    st.d_st_start
        .extend_from_slice(&rm.d_st[..nz.min(rm.d_st.len())]);
    st.d_st_start.resize(nz, 0.0);
    st.nu_st_start.clear();
    st.nu_st_start
        .extend_from_slice(&rm.nu_st[..nz.min(rm.nu_st.len())]);
    st.nu_st_start.resize(nz, 0.0);
    st.have_st_start_info = true;
}

/// `prepare_for_new_step`'s copy of the mesh and the ST start info before
/// remeshing (`star/private/evolve.f90:1859-1879`).
pub fn store_prev_mesh_st_info(model: &mut StarModel) {
    let nz = model.nz;
    let st = &mut model.aux.st_start;
    st.prev_mesh_dq.clear();
    st.prev_mesh_dq.extend_from_slice(&model.dq[..nz]);
    st.prev_mesh_nz = nz;
    if st.have_st_start_info {
        st.prev_mesh_d_st_start.clone_from(&st.d_st_start);
        st.prev_mesh_nu_st_start.clone_from(&st.nu_st_start);
        st.prev_mesh_have_st_start_info = true;
    } else {
        st.prev_mesh_have_st_start_info = false;
    }
}

/// Fortran `sign(a, b)` under `-fno-sign-zero`: `|a|` with the sign of `b`,
/// where `b = -0.0` counts as positive.
#[inline]
fn fsign(a: f64, b: f64) -> f64 {
    if b < 0.0 {
        -a.abs()
    } else {
        a.abs()
    }
}

/// `is_bad` (`utils_lib`): NaN or infinite.
#[inline]
fn is_bad(x: f64) -> bool {
    !x.is_finite()
}

/// `smooth` (`star/private/star_utils.f90:1969-1980`): a three-point filter
/// applied in place from the surface inwards, so each cell sees its already
/// smoothed outer neighbour.
pub(crate) fn smooth_in_place(dc: &mut [f64]) {
    let sz = dc.len();
    dc[0] = (3.0 * dc[0] + dc[1]) / 4.0;
    for k in 1..(sz - 1) {
        dc[k] = (dc[k - 1] + 2.0 * dc[k] + dc[k + 1]) / 4.0;
    }
    dc[sz - 1] = (dc[sz - 2] + 3.0 * dc[sz - 1]) / 4.0;
}

/// `safe_div_val(s, x, y)` (`star/private/star_utils.f90:4074-4089`) with the
/// default `lim = 1d-20`.
#[inline]
fn safe_div_val(x: f64, y: f64) -> f64 {
    if y.abs() < 1.0e-20 {
        0.0
    } else {
        x / y
    }
}

/// `grad_density` and `grad_temperature` from `set_grads`
/// (`star/private/hydro_vars.f90:1050-1076`).
pub(crate) fn set_grad_density_and_temperature(model: &mut StarModel) {
    let nz = model.nz;
    if nz < 2 {
        return;
    }
    model.aux.rotmix.resize(nz);
    let mut dlnp = vec![0.0_f64; nz];
    let mut dlnd = vec![0.0_f64; nz];
    let mut dlnt = vec![0.0_f64; nz];
    for k in 1..nz {
        dlnp[k] = model.ln_peos[k - 1] - model.ln_peos[k];
        dlnd[k] = model.lnd[k - 1] - model.lnd[k];
        dlnt[k] = model.ln_t[k - 1] - model.ln_t[k];
    }
    dlnp[0] = dlnp[1];
    dlnd[0] = dlnd[1];
    dlnt[0] = dlnt[1];
    smooth_in_place(&mut dlnp);
    smooth_in_place(&mut dlnd);
    smooth_in_place(&mut dlnt);
    let rm = &mut model.aux.rotmix;
    rm.grad_density[0] = 0.0;
    rm.grad_temperature[0] = 0.0;
    for k in 1..nz {
        if dlnp[k] >= 0.0 {
            rm.grad_density[k] = 0.0;
            rm.grad_temperature[k] = 0.0;
        } else {
            rm.grad_density[k] = safe_div_val(dlnd[k], dlnp[k]);
            rm.grad_temperature[k] = safe_div_val(dlnt[k], dlnp[k]);
        }
    }
    smooth_in_place(&mut rm.grad_density[..nz]);
    smooth_in_place(&mut rm.grad_temperature[..nz]);
}

/// `weighted_smoothing(dd, n, ns, preserve_sign, ddold)`
/// (`star/private/star_utils.f90:2777-2830`): binomial weights over
/// `2*ns + 1` cells, leaving the end cells unchanged.
pub(crate) fn weighted_smoothing(
    dd: &mut [f64],
    ns: usize,
    preserve_sign: bool,
) {
    let n = dd.len();
    let ddold = dd.to_vec();
    let nweight = ns;
    let mweight = 2 * nweight + 1;
    let mut weight = vec![0.0_f64; mweight];
    weight[0] = 1.0;
    for i in 1..mweight {
        // `do j = i+1, 2, -1` (1-based) => 0-based `j = i ..= 1`.
        for j in (1..=i).rev() {
            weight[j] += weight[j - 1];
        }
    }
    for i in 1..n.saturating_sub(1) {
        let mut sweight = 0.0_f64;
        dd[i] = 0.0;
        let v0 = ddold[i];
        // `do j = i, max(1, i-nweight), -1`
        let jlo = i.saturating_sub(nweight);
        for j in (jlo..=i).rev() {
            let k = j + nweight - i;
            if preserve_sign && v0 * ddold[j] <= 0.0 {
                break;
            }
            sweight += weight[k];
            dd[i] += ddold[j] * weight[k];
        }
        // `do j = i+1, min(n, i+nweight)`
        let jhi = (n - 1).min(i + nweight);
        for j in (i + 1)..=jhi {
            let k = j + nweight - i;
            if preserve_sign && v0 * ddold[j] <= 0.0 {
                break;
            }
            sweight += weight[k];
            dd[i] += ddold[j] * weight[k];
        }
        if sweight > 0.0 {
            let inv = 1.0 / sweight;
            dd[i] *= inv;
        }
    }
}

/// `smooth_for_rotation` (`rotation_mix_info.f90:1126-1134`).
fn smooth_for_rotation(v: &mut [f64], width: i64) {
    if width <= 0 {
        return;
    }
    weighted_smoothing(v, width as usize, false);
}

/// `zero_if_convective` (`rotation_mix_info.f90:1349-1357`).
fn zero_if_convective(mixing_type: &[MixingType], dc: &mut [f64]) {
    for (d, mt) in dc.iter_mut().zip(mixing_type) {
        if *mt == MixingType::ConvectiveMixing {
            *d = 0.0;
        }
    }
}

/// `zero_if_tiny` (`rotation_mix_info.f90:1360-1369`).
fn zero_if_tiny(clip_d_limit: f64, dc: &mut [f64]) {
    for d in dc.iter_mut() {
        if *d < clip_d_limit {
            *d = 0.0;
        }
    }
}

/// Face quantities built by `setup` (`rotation_mix_info.f90:431-724`).
struct Setup {
    csound: Vec<f64>,
    rho: Vec<f64>,
    t: Vec<f64>,
    cv: Vec<f64>,
    opacity: Vec<f64>,
    gradt_sub_grada: Vec<f64>,
    scale_height: Vec<f64>,
    omega: Vec<f64>,
    dr: Vec<f64>,
    drho_dr: Vec<f64>,
    drho_dr_ad: Vec<f64>,
    dr2omega: Vec<f64>,
    domega_dlnr: Vec<f64>,
    h_t: Vec<f64>,
    hj: Vec<f64>,
    dlnr_domega: Vec<f64>,
    t_dyn: Vec<f64>,
    ri_mu: Vec<f64>,
    ri_t: Vec<f64>,
    ve0: Vec<f64>,
    ve_mu: Vec<f64>,
    cp: Vec<f64>,
    abar: Vec<f64>,
    zbar: Vec<f64>,
    delta: Vec<f64>,
    n2: Vec<f64>,
    n2_mu: Vec<f64>,
    /// Electron-conduction opacity; `None` where upstream leaves `kap_cond`
    /// unset (`:656, 712-720`; #503).
    kap_cond: Option<Vec<f64>>,
}

#[allow(clippy::too_many_lines)]
fn setup(model: &mut StarModel) -> Result<Setup, StarError> {
    let nz = model.nz;
    let mw = &model.controls.mix_wind;
    let f_mu = mw.am_gradmu_factor;
    let r = &model.r;
    let m = &model.m;
    let lum = &model.lum;
    let j_rot = &model.j_rot;
    let gradt = &model.gradt;
    let grada = &model.aux.mixing.grada_face;
    let grav = &model.grav;
    let cgrav = &model.cgrav;
    let brunt = &model.aux.smoothed_brunt_b;
    let z = || vec![0.0_f64; nz];

    let (mut enu, mut eps_nuc, mut csound, mut rho, mut t, mut p) =
        (z(), z(), z(), z(), z(), z());
    let (mut cp, mut cv, mut chi_rho, mut abar, mut zbar, mut opacity) =
        (z(), z(), z(), z(), z(), z());
    let (mut gamma1, mut mu_alt, mut delta, mut l_neu) = (z(), z(), z(), z());

    // Interpolate by mass to the cell boundaries (`:476-532`).
    let mut enu00 = model.eps_nuc_neu_total[0] + model.non_nuc_neu[0];
    enu[0] = enu00;
    let mut eps_nuc00 = model.eps_nuc[0];
    eps_nuc[0] = eps_nuc00;
    csound[0] = model.csound[0];
    rho[0] = model.rho[0];
    t[0] = model.t[0];
    p[0] = model.peos[0];
    cp[0] = model.cp[0];
    cv[0] = model.cv[0];
    chi_rho[0] = model.chi_rho[0];
    abar[0] = model.abar[0];
    zbar[0] = model.zbar[0];
    opacity[0] = model.opacity[0];
    gamma1[0] = model.gamma1[0];
    mu_alt[0] = model.abar[0] / (1.0 + model.zbar[0]);
    delta[0] = model.chi_t[0] / model.chi_rho[0];
    l_neu[0] = enu00 * model.dm[0];
    for k in 1..nz {
        let alfa = model.dq[k - 1] / (model.dq[k - 1] + model.dq[k]);
        let beta = 1.0 - alfa;
        let enum1 = enu00;
        enu00 = model.eps_nuc_neu_total[k] + model.non_nuc_neu[k];
        enu[k] = alfa * enu00 + beta * enum1;
        let eps_nucm1 = eps_nuc00;
        eps_nuc00 = model.eps_nuc[k];
        eps_nuc[k] = alfa * eps_nuc00 + beta * eps_nucm1;
        csound[k] = alfa * model.csound[k] + beta * model.csound[k - 1];
        rho[k] = alfa * model.rho[k] + beta * model.rho[k - 1];
        t[k] = alfa * model.t[k] + beta * model.t[k - 1];
        p[k] = alfa * model.peos[k] + beta * model.peos[k - 1];
        cp[k] = alfa * model.cp[k] + beta * model.cp[k - 1];
        cv[k] = alfa * model.cv[k] + beta * model.cv[k - 1];
        chi_rho[k] = alfa * model.chi_rho[k] + beta * model.chi_rho[k - 1];
        abar[k] = alfa * model.abar[k] + beta * model.abar[k - 1];
        zbar[k] = alfa * model.zbar[k] + beta * model.zbar[k - 1];
        opacity[k] = alfa * model.opacity[k] + beta * model.opacity[k - 1];
        gamma1[k] = alfa * model.gamma1[k] + beta * model.gamma1[k - 1];
        mu_alt[k] = alfa * model.abar[k] / (1.0 + model.zbar[k])
            + beta * model.abar[k - 1] / (1.0 + model.zbar[k - 1]);
        delta[k] = alfa * model.chi_t[k] / model.chi_rho[k]
            + beta * model.chi_t[k - 1] / model.chi_rho[k - 1];
        l_neu[k] = enu00 * model.dm[k] + l_neu[k - 1];
    }
    let _ = &enu;

    let (mut gradt_sub_grada, mut scale_height, mut omega) = (z(), z(), z());
    for i in 0..nz {
        let g = gradt[i] - grada[i];
        gradt_sub_grada[i] = fsign(g.abs().max(1.0e-99), g);
        scale_height[i] = p[i] * r[i] * r[i] / (cgrav[i] * m[i] * rho[i]);
        let scale_height2 = (p[i] / cgrav[i]).sqrt() / rho[i];
        if scale_height2 < scale_height[i] {
            scale_height[i] = scale_height2;
        }
        omega[i] = model.omega[i];
    }

    // Differences at cell boundaries (`:546-571`).
    let (mut d_rho, mut dr, mut d_pressure, mut d_mu, mut d_j_rot, mut domega) =
        (z(), z(), z(), z(), z(), z());
    for i in 1..(nz - 1) {
        d_rho[i] = rho[i - 1] - rho[i];
        dr[i] = (0.5 * (r[i - 1] - r[i + 1])).max(1.0);
        d_pressure[i] = (p[i - 1] - p[i]).min(-1.0e-10);
        d_mu[i] = mu_alt[i - 1] - mu_alt[i];
        d_j_rot[i] = j_rot[i - 1] - j_rot[i];
        domega[i] = 0.5 * (omega[i - 1] - omega[i + 1]);
    }
    let _ = &d_mu;
    d_rho[0] = 0.0;
    d_rho[nz - 1] = 0.0;
    dr[0] = (r[0] - r[1]).max(1.0);
    dr[nz - 1] = r[nz - 1] - model.r_center;
    d_pressure[0] = 0.0;
    d_pressure[nz - 1] = 0.0;
    d_j_rot[0] = 0.0;
    d_j_rot[nz - 1] = 0.0;
    domega[0] = 0.0;
    domega[nz - 1] = 0.0;

    let (
        mut drho_dr,
        mut drho_dr_ad,
        mut dr2omega,
        mut domega_dlnr,
        mut h_t,
        mut hj,
    ) = (z(), z(), z(), z(), z(), z());
    for i in 1..(nz - 1) {
        drho_dr[i] = d_rho[i] / dr[i];
        drho_dr_ad[i] = rho[i] * d_pressure[i] / (p[i] * gamma1[i] * dr[i]);
        // d(r^2 omega)^2/dr using i = (2/3)*r^2
        dr2omega[i] = 4.5 * j_rot[i] * d_j_rot[i] / dr[i];
        domega_dlnr[i] = domega[i] * r[i] / dr[i];
        h_t[i] = if gradt[i] > 1.0e-20 {
            scale_height[i] / gradt[i]
        } else {
            scale_height[i]
        };
        hj[i] = if d_j_rot[i].abs() > 0.0 {
            j_rot[i] * dr[i] / d_j_rot[i]
        } else {
            0.0
        };
    }
    drho_dr[0] = 0.0;
    drho_dr[nz - 1] = 0.0;
    drho_dr_ad[0] = 0.0;
    drho_dr_ad[nz - 1] = 0.0;
    dr2omega[0] = 0.0;
    dr2omega[nz - 1] = 0.0;
    domega_dlnr[0] = 0.0;
    domega_dlnr[nz - 1] = 0.0;
    h_t[0] = h_t[1];
    h_t[nz - 1] = h_t[nz - 2];
    hj[0] = hj[1];
    hj[nz - 1] = hj[nz - 2];

    let (mut n2, mut n2_mu) = (z(), z());
    {
        let gd = &model.aux.rotmix.grad_density;
        let gtp = &model.aux.rotmix.grad_temperature;
        for i in 0..nz {
            let dlnrho_dlnp = gd[i];
            let dlnt_dlnp = gtp[i];
            n2[i] =
                -(grav[i] * (1.0 / gamma1[i] - dlnrho_dlnp) / scale_height[i]);
            n2_mu[i] = -(grav[i] / scale_height[i]
                * (1.0 / chi_rho[i] - delta[i] * dlnt_dlnp - dlnrho_dlnp));
        }
    }

    // Safe inverse of domega/dlnR (`:628-632`).
    let mut dlnr_domega = z();
    for i in 1..(nz - 1) {
        dlnr_domega[i] =
            fsign(1.0 / domega_dlnr[i].abs().max(1.0e-30), domega_dlnr[i]);
    }

    let mut omega_shear = z();
    for k in 1..(nz - 1) {
        let v = (omega[k - 1] - omega[k + 1]) / (r[k - 1] - r[k + 1])
            * (r[k] / omega[k]);
        // `max(1d-30, min(1d30, abs(...)))`; `clamp` would propagate a NaN.
        #[allow(clippy::manual_clamp)]
        {
            omega_shear[k] = v.abs().min(1.0e30).max(1.0e-30);
        }
    }

    // Timescales (`:642-646`).
    let (mut t_dyn, mut t_kh) = (z(), z());
    for i in 0..nz {
        t_dyn[i] = (r[i] * r[i] * r[i] / (cgrav[i] * m[i])).sqrt();
        t_kh[i] =
            cgrav[i] * m[i] * m[i] / (r[i] * (lum[i] + l_neu[i]).max(1.0));
    }

    // Richardson numbers (Heger 2000, eqn 20; `:649-662`).
    let (mut ri_t, mut ri_mu, mut ri) = (z(), z(), z());
    for i in 1..(nz - 1) {
        let ri0 = (rho[i] * delta[i] / p[i])
            * flexmesa_math::pow2(dlnr_domega[i] * grav[i]);
        ri_t[i] = ri0 * (-gradt_sub_grada[i]).max(0.0);
        ri_mu[i] = ri0 * f_mu * brunt[i];
    }
    for i in 0..nz {
        ri[i] = if n2[i] < 0.0 { 1.0 } else { ri_t[i] + ri_mu[i] };
    }

    // Dynamic viscosity (`:665-692`).
    let mut visc = z();
    for i in 0..nz {
        let rho6 = rho[i] * 1.0e-6;
        let gamma = 0.2275
            * zbar[i]
            * zbar[i]
            * flexmesa_math::pow(rho6 / abar[i], ONE_THIRD)
            * 1.0e8
            / t[i];
        let mu_e = abar[i] / zbar[i];
        let rm23 = flexmesa_math::pow(rho6 / mu_e, 2.0 / 3.0);
        let ctmp = 1.0 + 1.018 * rm23;
        let xi2 = (PI / 3.0).sqrt() * flexmesa_math::log(zbar[i]) / 3.0
            + 2.0 * flexmesa_math::log(1.32 + 2.33 / gamma.sqrt())
            - 0.475 * (1.0 + 2.036 * rm23) / ctmp
            + 0.276 * rm23 / ctmp;
        let ctmp = -0.016_321_227
            + 1.019_885_0 * flexmesa_math::pow(gamma, -1.921_797_0)
            + 0.024_113_535 * flexmesa_math::pow(gamma, 0.499_990_98);
        let mut dynvisc =
            5.53e3 * zbar[i] * flexmesa_math::pow(rho6, 5.0 / 6.0) * ctmp
                / flexmesa_math::pow(abar[i], ONE_THIRD);
        dynvisc += 4.0 * CRAD * flexmesa_math::pow4(t[i])
            / (15.0 * CLIGHT * opacity[i] * rho[i]);
        dynvisc +=
            1.893e6 * flexmesa_math::pow(rm23, 2.5) / (zbar[i] * ctmp * xi2);
        visc[i] = dynvisc / rho[i];
    }

    // Velocities for ES and GSF (Heger 2000, eqns 35-36; `:695-722`).
    let (mut ve0, mut ve_mu) = (z(), z());
    let mut kap_cond: Option<Vec<f64>> = None;
    if mw.d_es_factor > 0.0 || mw.d_gsf_factor > 0.0 {
        for i in 0..nz {
            let bracket_term = 2.0
                * r[i]
                * r[i]
                * (eps_nuc[i] / (1.0e-3 * L_SUN).max(lum[i].abs())
                    - 1.0 / (1.0e-3 * M_SUN).max(m[i]))
                - 3.0 / (PI4 * rho[i] * (1.0e-3 * R_SUN).max(r[i]));
            if gradt_sub_grada[i].abs() < 1.0e-50 {
                ve0[i] = 1.0e99;
                ve_mu[i] = 1.0e99;
            } else {
                let denom = -gradt_sub_grada[i]
                    * delta[i]
                    * flexmesa_math::pow2(cgrav[i] * m[i]);
                if denom.abs() < 1.0e-50 || is_bad(denom) {
                    ve0[i] = 1.0e99;
                } else {
                    ve0[i] = grada[i]
                        * omega[i]
                        * omega[i]
                        * r[i]
                        * r[i]
                        * r[i]
                        * lum[i]
                        * bracket_term
                        / denom;
                }
                ve_mu[i] = (scale_height[i] / t_kh[i]) * (f_mu * brunt[i])
                    / gradt_sub_grada[i];
            }
            if is_bad(ve0[i]) {
                ve0[i] = 1.0e99;
            }
            if is_bad(ve_mu[i]) {
                ve_mu[i] = 1.0e99;
            }
        }
        // Conductive opacities for ST (`:713-721`).
        if mw.d_st_factor > 0.0 || mw.am_nu_factor > 0.0 {
            let mut kc = z();
            for i in 0..nz {
                let (kap, _, _) = flexmesa_kap::conduction::do_electron_conduction(
                    &model.kap_config,
                    zbar[i],
                    flexmesa_math::log10(rho[i]),
                    flexmesa_math::log10(t[i]),
                )
                .map_err(|e| {
                    StarError::MicrophysicsError(format!(
                        "set_rotation_mixing_info: kap_get_elect_cond_opacity failed at face {}: {e} (star/private/rotation_mix_info.f90:715-719)",
                        i + 1
                    ))
                })?;
                kc[i] = kap;
            }
            kap_cond = Some(kc);
        }
    }

    let rm = &mut model.aux.rotmix;
    rm.d_visc[..nz].copy_from_slice(&visc);
    rm.domega_dlnr[..nz].copy_from_slice(&domega_dlnr);
    rm.omega_shear[..nz].copy_from_slice(&omega_shear);
    rm.richardson_number[..nz].copy_from_slice(&ri);

    Ok(Setup {
        csound,
        rho,
        t,
        cv,
        opacity,
        gradt_sub_grada,
        scale_height,
        omega,
        dr,
        drho_dr,
        drho_dr_ad,
        dr2omega,
        domega_dlnr,
        h_t,
        hj,
        dlnr_domega,
        t_dyn,
        ri_mu,
        ri_t,
        ve0,
        ve_mu,
        cp,
        abar,
        zbar,
        delta,
        n2,
        n2_mu,
        kap_cond,
    })
}

/// Walks the unstable blocks from the centre outwards exactly as upstream's
/// `kbot` / `ktop` loops (`rotation_mix_info.f90:741-754`), calling
/// `body(ktop, kbot)` with 0-based inclusive bounds for every block of at
/// least two faces.
fn for_each_block(unstable: &[bool], mut body: impl FnMut(usize, usize)) {
    let nz = unstable.len();
    // 1-based upstream: `kbot = nz; do i = nz-1, 1, -1`.
    let mut kbot = nz;
    for i in (1..nz).rev() {
        let ui = unstable[i - 1];
        let uip1 = unstable[i];
        if ui && !uip1 {
            kbot = i;
        }
        if uip1 && (i == 1 || !ui) && kbot > 1 {
            let ktop = if ui { i } else { i + 1 };
            if ktop >= kbot {
                continue;
            }
            body(ktop - 1, kbot - 1);
        }
    }
}

fn set_d_dsi(model: &mut StarModel, su: &Setup) -> Result<(), StarError> {
    let nz = model.nz;
    let ri = model.aux.rotmix.richardson_number.clone();
    let unstable: Vec<bool> = (0..nz)
        .map(|i| ri[i] < RI_CRIT && su.gradt_sub_grada[i] < 0.0)
        .collect();
    let r = &model.r;
    let d = &mut model.aux.rotmix.d_dsi;
    let mut err = None;
    for_each_block(&unstable, |ktop, kbot| {
        let instability_height = r[ktop] - r[kbot];
        for k in ktop..=kbot {
            if !unstable[k] {
                err = Some(k);
            }
            let height = instability_height.min(su.scale_height[k]);
            let dd = height * height / su.t_dyn[k];
            d[k] = dd.min(su.scale_height[k] * su.csound[k]);
        }
    });
    if let Some(k) = err {
        return Err(StarError::ConvergenceFailure(format!(
            "set_D_DSI: face {} inside an unstable block is stable (star/private/rotation_mix_info.f90:757-768)",
            k + 1
        )));
    }
    Ok(())
}

fn set_d_sh(model: &mut StarModel, su: &Setup) {
    let nz = model.nz;
    let r = &model.r;
    let grav = &model.grav;
    let d = &mut model.aux.rotmix.d_sh;
    let mut unstable = vec![false; nz];
    for i in 0..nz {
        let dd = grav[i] / su.rho[i] * (su.drho_dr_ad[i] - su.drho_dr[i])
            + su.dr2omega[i] / (r[i] * r[i] * r[i]);
        if dd < 0.0 {
            unstable[i] = true;
            d[i] = dd;
        } else {
            d[i] = 0.0;
        }
    }
    for_each_block(&unstable, |ktop, kbot| {
        let instability_height = r[ktop] - r[kbot];
        for k in ktop..=kbot {
            let height = instability_height.min(su.scale_height[k]);
            let dd = flexmesa_math::pow2(height * d[k] * r[k] / grav[k])
                / su.t_dyn[k];
            d[k] = dd.min(su.scale_height[k] * su.csound[k]);
        }
    });
}

fn set_d_ssi(model: &mut StarModel, su: &Setup) -> Result<(), StarError> {
    let nz = model.nz;
    let r = &model.r;
    let visc = model.aux.rotmix.d_visc.clone();
    let mut unstable = vec![false; nz];
    let (mut ris_1, mut ris_2) = (vec![0.0_f64; nz], vec![0.0_f64; nz]);
    for i in 0..nz {
        let radcon = 4.0 * CRAD * CLIGHT * su.t[i] * su.t[i] * su.t[i]
            / (3.0 * su.opacity[i] * su.rho[i]);
        let dynvisc = visc[i] * su.rho[i];
        let prandtl = dynvisc * su.cv[i] / radcon;
        ris_1[i] = 0.125 * R_CRIT * prandtl * su.ri_t[i];
        if ris_1[i] <= RI_CRIT {
            ris_2[i] = su.ri_mu[i];
            if ris_2[i] <= RI_CRIT {
                unstable[i] = true;
            }
        } else {
            ris_2[i] = 0.0;
        }
    }
    let d = &mut model.aux.rotmix.d_ssi;
    let mut v_ssi = vec![0.0_f64; nz];
    let mut h_ssi = vec![0.0_f64; nz];
    let mut bad = None;
    for_each_block(&unstable, |ktop, kbot| {
        for k in ktop..=kbot {
            v_ssi[k] = (visc[k] / R_CRIT * su.domega_dlnr[k].abs()).sqrt();
        }
        h_ssi[kbot] = v_ssi[kbot] * (r[kbot - 1] - r[kbot])
            / (v_ssi[kbot - 1] - v_ssi[kbot]).abs().max(1.0e-99);
        for k in ((ktop + 1)..kbot).rev() {
            h_ssi[k] = v_ssi[k] * su.dr[k]
                / (0.5 * (v_ssi[k + 1] - v_ssi[k - 1]).abs()).max(1.0e-99);
        }
        h_ssi[ktop] = v_ssi[ktop] * (r[ktop] - r[ktop + 1])
            / (v_ssi[ktop] - v_ssi[ktop + 1]).abs().max(1.0e-99);
        for k in ktop..=kbot {
            h_ssi[k] = h_ssi[k].min(su.scale_height[k]);
            v_ssi[k] = v_ssi[k].min(su.csound[k]);
            let dd = h_ssi[k]
                * v_ssi[k]
                * flexmesa_math::pow2(
                    1.0 - (ris_1[k].max(ris_2[k]) / RI_CRIT).max(0.0),
                );
            d[k] = dd.min(su.scale_height[k] * su.csound[k]);
            if d[k] > 1.0e100 {
                bad = Some(k);
            }
        }
    });
    if let Some(k) = bad {
        return Err(StarError::ConvergenceFailure(format!(
            "set_D_SSI: D_SSI > 1e100 at face {} (star/private/rotation_mix_info.f90:899-903)",
            k + 1
        )));
    }
    Ok(())
}

fn set_d_es(model: &mut StarModel, su: &Setup) {
    let nz = model.nz;
    let r = &model.r;
    let mut unstable = vec![false; nz];
    let mut v_es = vec![0.0_f64; nz];
    for i in 0..nz {
        let v = su.ve0[i].abs() - su.ve_mu[i].abs();
        if v > 0.0 {
            unstable[i] = true;
            v_es[i] = v;
        } else {
            v_es[i] = 0.0;
        }
    }
    let d = &mut model.aux.rotmix.d_es;
    let mut h_es = vec![0.0_f64; nz];
    for_each_block(&unstable, |ktop, kbot| {
        let instability_height = r[ktop] - r[kbot];
        h_es[kbot] = v_es[kbot] * (r[kbot - 1] - r[kbot])
            / (v_es[kbot - 1] - v_es[kbot]).abs().max(1.0e-99);
        for k in ((ktop + 1)..kbot).rev() {
            h_es[k] = v_es[k] * su.dr[k]
                / (0.5 * (v_es[k + 1] - v_es[k - 1]).abs()).max(1.0e-99);
        }
        h_es[ktop] = v_es[ktop] * (r[ktop] - r[ktop + 1])
            / (v_es[ktop] - v_es[ktop + 1]).abs().max(1.0e-99);
        for k in ktop..=kbot {
            h_es[k] = instability_height.min(h_es[k]).min(su.scale_height[k]);
            v_es[k] = v_es[k].min(su.csound[k]);
            let dd = h_es[k] * v_es[k];
            d[k] = dd.min(su.scale_height[k] * su.csound[k]);
        }
    });
}

fn set_d_gsf(model: &mut StarModel, su: &Setup) -> Result<(), StarError> {
    let nz = model.nz;
    let r = &model.r;
    let r_center = model.r_center;
    let mut unstable = vec![false; nz];
    let mut v_gsf = vec![0.0_f64; nz];
    for i in 0..nz {
        let v = if su.hj[i].abs() > 0.0 {
            su.ve0[i] * 2.0 * su.h_t[i] * r[i]
                / (su.hj[i] * su.hj[i])
                / (1.0 + 2.0 * su.omega[i] * su.dlnr_domega[i])
        } else {
            0.0
        };
        if is_bad(v) {
            return Err(StarError::ConvergenceFailure(format!(
                "set_D_GSF: bad v at face {} (star/private/rotation_mix_info.f90:986-991)",
                i + 1
            )));
        }
        let v_diff = v.abs() - su.ve_mu[i].abs();
        if v_diff > 0.0 {
            unstable[i] = true;
            v_gsf[i] = v_diff;
        } else {
            v_gsf[i] = 0.0;
        }
    }
    let d = &mut model.aux.rotmix.d_gsf;
    let mut h_gsf = vec![0.0_f64; nz];
    for_each_block(&unstable, |ktop, kbot| {
        let instability_height = r[ktop] - r[kbot];
        // `kbot == 1` (1-based) cannot occur: the block loop requires
        // `kbot > 1` (`:1027`).
        if kbot == 0 {
            h_gsf[kbot] = v_gsf[kbot] * (r_center - r[kbot])
                / v_gsf[kbot].abs().max(1.0e-99);
        } else {
            h_gsf[kbot] = v_gsf[kbot] * (r[kbot - 1] - r[kbot])
                / (v_gsf[kbot - 1] - v_gsf[kbot]).abs().max(1.0e-99);
        }
        // `do k = max(2,kbot-1), min(nz-1,ktop+1), -1` (1-based; with
        // 0-based `kbot`, upstream's `kbot-1` is `kbot`).
        let k_hi = kbot.max(2) - 1;
        let k_lo = (ktop + 2).min(nz - 1) - 1;
        if k_hi >= k_lo {
            for k in (k_lo..=k_hi).rev() {
                h_gsf[k] = v_gsf[k] * su.dr[k]
                    / (0.5 * (v_gsf[k + 1] - v_gsf[k - 1]).abs()).max(1.0e-99);
            }
        }
        if ktop == nz - 1 {
            h_gsf[ktop] = v_gsf[ktop] * (r[ktop] - r_center)
                / v_gsf[ktop].abs().max(1.0e-99);
        } else {
            h_gsf[ktop] = v_gsf[ktop] * (r[ktop] - r[ktop + 1])
                / (v_gsf[ktop] - v_gsf[ktop + 1]).abs().max(1.0e-99);
        }
        for k in ktop..=kbot {
            h_gsf[k] = instability_height.min(h_gsf[k]).min(su.scale_height[k]);
            v_gsf[k] = v_gsf[k].min(su.csound[k]);
            let dd = h_gsf[k] * v_gsf[k];
            d[k] = dd.min(su.scale_height[k] * su.csound[k]);
        }
    });
    Ok(())
}

/// `calc_eta` (`star/private/magnetic_diffusion.f90:38-41`): magnetic
/// diffusivity (cm^2 s^-1) from the electrical conductivity `sig` (s^-1).
fn calc_eta(sig: f64) -> f64 {
    (CLIGHT * CLIGHT / (4.0 * PI)) / sig
}

/// `sige1` (`magnetic_diffusion.f90:88-100`).
fn sige1(z: f64, t: f64, xgamma: f64) -> f64 {
    let f = if t >= 4.2e5 { (4.2e5 / t).sqrt() } else { 1.0 };
    let xlambda =
        (3.0 * z * z * z).sqrt() * flexmesa_math::pow(xgamma, -1.5) * f + 1.0;
    let mut etan =
        3.0e11 * z * flexmesa_math::log(xlambda) * flexmesa_math::pow(t, -1.5);
    etan /= 1.0
        - 1.20487
            * flexmesa_math::exp(-1.0576 * flexmesa_math::pow(z, 0.347044));
    CLIGHT * CLIGHT / (PI4 * etan)
}

/// `sige2` (`magnetic_diffusion.f90:110-113`).
fn sige2(t: f64, rho: f64, kap_cond: f64) -> f64 {
    1.11e9 * t * t / (rho * kap_cond)
}

/// `sige3` (`magnetic_diffusion.f90:123-133`).
fn sige3(z: f64, t: f64, xgamma: f64) -> f64 {
    let rme = 8.5646e-23 * t * t * t * xgamma * xgamma * xgamma
        / flexmesa_math::pow5(z);
    let rm23 = flexmesa_math::pow(rme, 2.0 / 3.0);
    let ctmp = 1.0 + 1.018 * rm23;
    let xi = (PI / 3.0).sqrt() * flexmesa_math::log(z) / 3.0
        + 2.0 * flexmesa_math::log(1.32 + 2.33 / xgamma.sqrt()) / 3.0
        - 0.484 * rm23 / ctmp;
    8.630e21 * rme / (z * ctmp * xi)
}

/// `calc_sige` (`magnetic_diffusion.f90:53-78`): electrical conductivity
/// (s^-1) after S.-C. Yoon (2003).
fn calc_sige(abar: f64, zbar: f64, rho: f64, t: f64, kap_cond: f64) -> f64 {
    let gamma = 0.2275
        * flexmesa_math::pow2(zbar)
        * flexmesa_math::pow(rho * 1.0e-6 / abar, ONE_THIRD)
        * 1.0e8
        / t;
    let xlg = flexmesa_math::log10(gamma);
    // `alpha` (`:58`) is computed upstream but never used.
    if xlg < -1.5 {
        sige1(zbar, t, gamma)
    } else if (-1.5..=0.0).contains(&xlg) {
        let xxx = (xlg + 0.75) * 4.0 / 3.0;
        let ffff = 0.25 * (2.0 - 3.0 * xxx + flexmesa_math::pow3(xxx));
        let xsig1 = sige1(zbar, t, gamma);
        let xsig2 = sige2(t, rho, kap_cond);
        (1.0 - ffff) * xsig2 + ffff * xsig1
    } else if xlg > 0.0 && xlg < 0.5 {
        sige2(t, rho, kap_cond)
    } else if (0.5..1.0).contains(&xlg) {
        let xxx = (xlg - 0.75) * 4.0;
        let ffff = 0.25 * (2.0 - 3.0 * xxx + flexmesa_math::pow3(xxx));
        let xsig2 = sige2(t, rho, kap_cond);
        let xsig3 = sige3(zbar, t, gamma);
        (1.0 - ffff) * xsig3 + ffff * xsig2
    } else {
        sige3(zbar, t, gamma)
    }
}

/// `set_ST` (`star/private/rotation_mix_info.f90:1137-1346`): Spruit-Tayler
/// dynamo diffusivity `D_ST` and viscosity `nu_ST`, with S.-C. Yoon's 2003
/// modifications.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn set_st(model: &mut StarModel, su: &Setup) -> Result<(), StarError> {
    let nz = model.nz;
    let Some(kap_cond) = su.kap_cond.as_ref() else {
        // `kap_cond` is filled only when `D_ES_factor > 0 .or. D_GSF_factor > 0`
        // (`:656, 712-720`); otherwise `set_ST` reads an unset array at `:1182`
        // (deferred upstream bug #503; DECISIONS #43).
        return Err(StarError::InvalidInput(String::from(
            "set_ST: kap_cond is unset upstream unless D_ES_factor or D_GSF_factor > 0 (star/private/rotation_mix_info.f90:656, 712-720, 1182); deferred upstream bug #503",
        )));
    };
    let r = &model.r;
    let lum = &model.lum;
    let grav = &model.grav;
    let mixing_type = &model.mixing_type;
    let rm = &mut model.aux.rotmix;
    rm.d_st[..nz].fill(0.0);
    rm.nu_st[..nz].fill(0.0);
    rm.dynamo_b_r[..nz].fill(0.0);
    rm.dynamo_b_phi[..nz].fill(0.0);
    let (xmagfmu, xmagft, xmagfdif, xmagfnu) = (1.0, 1.0, 1.0, 1.0);
    let (mut xmageta0, mut xmageta1, mut xmagetam) =
        (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut xmagq0, mut xmagq1, mut xmagqm) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut xmags0, mut xmags1, mut xmagsm) = (0.0_f64, 0.0_f64, 0.0_f64);
    let omega = &su.omega;
    let (rho, t) = (&su.rho, &su.t);
    for k in 1..(nz - 1) {
        let xkap = 16.0 * flexmesa_const::BOLTZ_SIGMA * t[k] * t[k] * t[k]
            / (3.0 * su.opacity[k] * rho[k] * rho[k] * su.cp[k]);
        let xsig = calc_sige(su.abar[k], su.zbar[k], rho[k], t[k], kap_cond[k]);
        let xeta = calc_eta(xsig);
        let xmagn = su.n2[k];
        let xmagnmu = su.n2_mu[k];
        let xmagnt = xmagn - xmagnmu;
        let xmagw = omega[k].abs();
        let xmagdn = rho[k];
        let xmagtn = t[k];
        let xmagrn = r[k];
        let dlnomega_dlnr = (omega[k - 1] - omega[k + 1])
            / (r[k - 1] - r[k + 1])
            * (r[k] / omega[k]);
        // `max(1d-30, min(1d30, abs(...)))`; `clamp` would propagate a NaN.
        #[allow(clippy::manual_clamp)]
        let xmagq = dlnomega_dlnr.abs().min(1.0e30).max(1.0e-30);
        let xmager2w = xeta / (r[k] * r[k] * omega[k].abs());
        if xmagnmu > 0.0 {
            let xmagnn = xmagnmu * xmagfmu;
            let xmagwn = xmagw / xmagnn.sqrt();
            xmagq0 = flexmesa_math::pow(xmagwn, -1.5)
                * flexmesa_math::pow(xmager2w, 0.25);
            xmags0 = xmagdn
                * xmagw
                * xmagw
                * xmagrn
                * xmagrn
                * flexmesa_math::pow3(xmagq)
                * flexmesa_math::pow4(xmagwn);
            if is_bad(xmags0) {
                return Err(StarError::ConvergenceFailure(format!(
                    "set_ST: bad xmags0 at face {} (star/private/rotation_mix_info.f90:1203-1218)",
                    k + 1
                )));
            }
            xmageta0 = flexmesa_math::pow4(xmagq)
                * flexmesa_math::pow6(xmagwn)
                * xmagrn
                * xmagrn
                * xmagw;
        }
        if xmagnt > 0.0 {
            let xmagnn = xmagnt * xmagft;
            let xmagwn = xmagw / xmagnn.sqrt();
            let xmagkr2n = xkap / (xmagrn * xmagrn * xmagnn.sqrt());
            xmagq1 = if xmagwn > 1.0e-42 {
                flexmesa_math::pow(
                    xmager2w * xmagwn * flexmesa_math::pow3(xeta / xkap)
                        / flexmesa_math::pow7(xmagwn),
                    0.25,
                )
            } else {
                0.0
            };
            let xmags1a = xmagdn
                * flexmesa_math::pow2(xmagw * xmagrn)
                * xmagq
                * (xmagwn * xmagkr2n).sqrt();
            let xmags1b = xmagdn
                * flexmesa_math::pow2(xmagw)
                * flexmesa_math::pow2(xmagrn)
                * flexmesa_math::pow3(xmagq)
                * flexmesa_math::pow4(xmagwn);
            xmags1 = xmags1a.max(xmags1b);
            let xmageta1a = xmagrn
                * xmagrn
                * xmagw
                * xmagq
                * flexmesa_math::pow(xmagwn * xmagkr2n, 0.75);
            let xmageta1b = flexmesa_math::pow4(xmagq)
                * flexmesa_math::pow6(xmagwn)
                * flexmesa_math::pow2(xmagrn)
                * xmagw;
            xmageta1 = xmageta1a.max(xmageta1b);
        }
        if xmagnt > 0.0 && xmagnmu > 0.0 && xmags0 + xmags1 != 0.0 {
            xmagsm = xmags0 * xmags1 / (xmags0 + xmags1);
            if is_bad(xmagsm) {
                return Err(StarError::ConvergenceFailure(format!(
                    "set_ST: bad xmagsm at face {} (star/private/rotation_mix_info.f90:1250-1259)",
                    k + 1
                )));
            }
            xmagqm = xmagq0 + xmagq1;
            xmagetam = xmageta0 * xmageta1 / (xmageta0 + xmageta1);
        } else if xmagnt <= 0.0 && xmagnmu > 0.0 {
            xmagsm = xmags0;
            xmagqm = xmagq0;
            xmagetam = xmageta0;
        } else if xmagnt > 0.0 && xmagnmu <= 0.0 {
            xmagsm = xmags1;
            xmagqm = xmagq1;
            xmagetam = xmageta1;
        } else if xmagnt <= 0.0 && xmagnmu <= 0.0 {
            xmagsm = 0.0;
            xmagqm = 0.0;
            xmagetam = 0.0;
        }
        if xmagn < 0.0 {
            xmagsm = 0.0;
            xmagqm = 0.0;
            xmagetam = 0.0;
        }
        if mixing_type[k] == MixingType::ConvectiveMixing && xmagn > 0.0 {
            xmagsm = 0.0;
            xmagqm = 0.0;
            xmagetam = 0.0;
        }
        if mixing_type[k] == MixingType::OvershootMixing {
            xmagsm = 0.0;
            xmagqm = 0.0;
            xmagetam = 0.0;
        }
        let xmagsf = 1.0 - 1.0_f64.min(xmagqm / xmagq);
        let xmags = xmagsf * xmagsm;
        let mut xmagnu = xmags / (xmagw * xmagq * xmagdn);
        let xmagdif = xmagsf * xmagetam;
        // Semiconvective regions (`:1307-1312`).
        if xmagnt <= 0.0 && xmagnmu > 0.0 && xmagn > 0.0 {
            xmagnu = (xmagnu
                * su.scale_height[k]
                * ONE_THIRD
                * flexmesa_math::pow(
                    grav[k]
                        * su.delta[k]
                        * su.scale_height[k]
                        * 0.0_f64.max(lum[k])
                        / (64.0
                            * PI
                            * xmagdn
                            * su.cp[k]
                            * xmagtn
                            * xmagrn
                            * xmagrn),
                    ONE_THIRD,
                ))
            .sqrt();
        }
        rm.d_st[k] = xmagdif.min(xmagnu) * xmagfdif;
        rm.nu_st[k] = xmagnu * xmagfnu;
        if is_bad(rm.d_st[k]) {
            return Err(StarError::ConvergenceFailure(format!(
                "set_ST: bad D_ST at face {} (star/private/rotation_mix_info.f90:1315-1334)",
                k + 1
            )));
        }
    }
    rm.d_st[0] = rm.d_st[1];
    rm.d_st[nz - 1] = rm.d_st[nz - 2];
    rm.nu_st[0] = rm.nu_st[1];
    rm.nu_st[nz - 1] = rm.nu_st[nz - 2];
    Ok(())
}

/// The `i_ST` branch of `set_rotation_mixing_info`
/// (`rotation_mix_info.f90:151-209`): `set_ST`, smoothing, the time
/// smoothing against the previous step, the dynamo fields, and the
/// convective and tiny clips.
fn do_st(model: &mut StarModel, su: &Setup) -> Result<(), StarError> {
    let nz = model.nz;
    set_st(model, su)?;
    let mw = model.controls.mix_wind.clone();
    {
        let rm = &mut model.aux.rotmix;
        smooth_for_rotation(&mut rm.d_st[..nz], mw.smooth_d_st);
        smooth_for_rotation(&mut rm.nu_st[..nz], mw.smooth_nu_st);
    }
    // Time smoothing for ST only (`:162-189`).
    let st = &model.aux.st_start;
    if st.prev_mesh_have_st_start_info
        && !model.aux.doing_relax
        && !model.doing_first_model_of_run
        && mw.st_angsmt > 0.0
    {
        let angsml = mw.st_angsml;
        let angsmt = mw.st_angsmt;
        let mstar_old = model.mstar_old;
        let prev_dq = &st.prev_mesh_dq;
        let prev_nz = st.prev_mesh_nz;
        let mut k0 = 1_usize; // 1-based
        let mut prev_out_q = 0.0_f64;
        let mut prev_q = 1.0_f64;
        let rm = &mut model.aux.rotmix;
        for k in 1..(nz - 1) {
            if model.m[k] > mstar_old {
                continue;
            }
            while k0 < prev_nz {
                if model.m[k] / mstar_old > prev_q {
                    break;
                }
                prev_out_q += prev_dq[k0 - 1];
                prev_q = 1.0 - prev_out_q;
                k0 += 1;
            }
            if model.m[k] / mstar_old < prev_q {
                break;
            }
            if k0 < 2 {
                // `prev_mesh_dq(k0-1)` with `k0 = 1` reads index 0, outside
                // the array (`rotation_mix_info.f90:191-215`; deferred upstream
                // bug #504; DECISIONS #43).
                return Err(StarError::InvalidInput(String::from(
                    "set_rotation_mixing_info: ST time smoothing would read prev_mesh_dq(0) (star/private/rotation_mix_info.f90:191-215); deferred upstream bug #504",
                )));
            }
            let prev_out_q_m1 = prev_out_q - prev_dq[k0 - 2];
            let prev_q_m1 = 1.0 - prev_out_q_m1;
            let alfa = (model.m[k] / mstar_old - prev_q) / (prev_q_m1 - prev_q);
            let d_st_old = (1.0 - alfa) * st.prev_mesh_d_st_start[k0 - 1]
                + alfa * st.prev_mesh_d_st_start[k0 - 2];
            let nu_st_old = (1.0 - alfa) * st.prev_mesh_nu_st_start[k0 - 1]
                + alfa * st.prev_mesh_nu_st_start[k0 - 2];
            let dgtau = angsml
                * (model.r[k] - model.r[k + 1])
                * (model.r[k - 1] * model.r[k])
                / model.dt;
            rm.d_st[k] =
                (d_st_old / (1.0 + angsmt))
                    .max(rm.d_st[k].min(
                        (d_st_old * (1.0 + angsmt)).max(d_st_old + dgtau),
                    ));
            rm.nu_st[k] = (nu_st_old / (1.0 + angsmt)).max(
                rm.nu_st[k]
                    .min((nu_st_old * (1.0 + angsmt)).max(nu_st_old + dgtau)),
            );
        }
    }
    // Dynamo B_r and B_phi (Heger et al. 2005, eqns 11-12; `:191-197`).
    {
        let rm = &mut model.aux.rotmix;
        for k in 0..nz {
            let q = rm.omega_shear[k];
            rm.dynamo_b_r[k] = flexmesa_math::pow(
                flexmesa_math::pow2(
                    PI4 * su.rho[k] * rm.nu_st[k] * q / model.r[k],
                ) * su.omega[k].abs()
                    * rm.nu_st[k],
                0.25,
            );
            rm.dynamo_b_phi[k] = flexmesa_math::pow(
                flexmesa_math::pow2(
                    PI4 * su.rho[k] * su.omega[k] * q * model.r[k],
                ) * su.omega[k].abs()
                    * rm.nu_st[k],
                0.25,
            );
        }
    }
    let mixing_type = model.mixing_type.clone();
    let rm = &mut model.aux.rotmix;
    if mw.skip_rotation_in_convection_zones {
        zero_if_convective(&mixing_type, &mut rm.d_st[..nz]);
        zero_if_convective(&mixing_type, &mut rm.nu_st[..nz]);
        zero_if_convective(&mixing_type, &mut rm.dynamo_b_r[..nz]);
        zero_if_convective(&mixing_type, &mut rm.dynamo_b_phi[..nz]);
    }
    zero_if_tiny(mw.clip_d_limit, &mut rm.d_st[..nz]);
    zero_if_tiny(mw.clip_d_limit, &mut rm.nu_st[..nz]);
    Ok(())
}

/// `set_rotation_mixing_info` (`star/private/rotation_mix_info.f90:47-332`).
///
/// # Errors
/// Returns [`StarError`] for the upstream `mesa_error` paths and for a
/// failed conductive-opacity evaluation.
pub fn set_rotation_mixing_info(
    model: &mut StarModel,
) -> Result<(), StarError> {
    let nz = model.nz;
    model.aux.rotmix.resize(nz);
    {
        let rm = &mut model.aux.rotmix;
        for v in [
            &mut rm.d_visc,
            &mut rm.d_dsi,
            &mut rm.d_sh,
            &mut rm.d_ssi,
            &mut rm.d_es,
            &mut rm.d_gsf,
            &mut rm.d_st,
            &mut rm.nu_st,
            &mut rm.dynamo_b_r,
            &mut rm.dynamo_b_phi,
            &mut rm.omega_shear,
        ] {
            v[..nz].fill(0.0);
        }
    }
    if model.omega[..nz].iter().all(|&w| w == 0.0) {
        return Ok(());
    }
    if nz < 3 {
        return Err(StarError::InvalidInput(String::from(
            "set_rotation_mixing_info needs nz >= 3 (star/private/rotation_mix_info.f90:431-724)",
        )));
    }
    let su = setup(model)?;
    let mw = model.controls.mix_wind.clone();
    let skip_conv = mw.skip_rotation_in_convection_zones;
    let clip = mw.clip_d_limit;
    let mixing_type = model.mixing_type.clone();

    // `i_DSI` (`:96-106`).
    if mw.d_dsi_factor > 0.0 || mw.am_nu_dsi_factor > 0.0 {
        set_d_dsi(model, &su)?;
        let d = &mut model.aux.rotmix.d_dsi[..nz];
        smooth_for_rotation(d, mw.smooth_d_dsi);
        if skip_conv {
            zero_if_convective(&mixing_type, d);
        }
        zero_if_tiny(clip, d);
    }
    // `i_SH` (`:107-117`).
    if mw.d_sh_factor > 0.0 || mw.am_nu_sh_factor > 0.0 {
        set_d_sh(model, &su);
        let d = &mut model.aux.rotmix.d_sh[..nz];
        smooth_for_rotation(d, mw.smooth_d_sh);
        if skip_conv {
            zero_if_convective(&mixing_type, d);
        }
        zero_if_tiny(clip, d);
    }
    // `i_SSI` (`:118-128`).
    if mw.d_ssi_factor > 0.0 || mw.am_nu_ssi_factor > 0.0 {
        set_d_ssi(model, &su)?;
        let d = &mut model.aux.rotmix.d_ssi[..nz];
        smooth_for_rotation(d, mw.smooth_d_ssi);
        if skip_conv {
            zero_if_convective(&mixing_type, d);
        }
        zero_if_tiny(clip, d);
    }
    // `i_ES` (`:129-139`).
    if mw.d_es_factor > 0.0 || mw.am_nu_es_factor > 0.0 {
        set_d_es(model, &su);
        let d = &mut model.aux.rotmix.d_es[..nz];
        smooth_for_rotation(d, mw.smooth_d_es);
        if skip_conv {
            zero_if_convective(&mixing_type, d);
        }
        zero_if_tiny(clip, d);
    }
    // `i_GSF` (`:140-150`).
    if mw.d_gsf_factor > 0.0 || mw.am_nu_gsf_factor > 0.0 {
        set_d_gsf(model, &su)?;
        let d = &mut model.aux.rotmix.d_gsf[..nz];
        smooth_for_rotation(d, mw.smooth_d_gsf);
        if skip_conv {
            zero_if_convective(&mixing_type, d);
        }
        zero_if_tiny(clip, d);
    }
    // `i_ST` (`:151-209`).
    if mw.d_st_factor > 0.0 || mw.am_nu_st_factor > 0.0 {
        do_st(model, &su)?;
    }
    Ok(())
}
