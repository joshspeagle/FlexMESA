// SPDX-License-Identifier: GPL-3.0-or-later
//! OSC pulsation data: `get_osc_data` (`star/private/pulse_osc.f90:46-552`),
//! transcribed in upstream order of operations (#491).
//!
//! Returns `global_data(15)` and `point_data(36, nn)` from the surface
//! inwards, as [`crate::pulse_write::write_osc_data`] writes them.

use flexmesa_const::{FOUR_THIRDS, PI, RSUN};

use crate::pulse_fgong::{eval_center_d2, PulseDataInput};
use crate::pulse_gyre_data::{
    eval_center, eval_center_rho, eval_face, eval_face_a_ast, eval_face_rho,
    set_segment_indices, GyreDataError,
};
use crate::pulse_write::{OSC_IABUND, OSC_ICONST, OSC_IVAR};

fn at(v: &[f64], k: usize) -> f64 {
    v[k - 1]
}

/// `get_osc_data(id, add_center_point, keep_surface_point, add_atmosphere,
/// global_data, point_data, ierr)` (`star/private/pulse_osc.f90:46-552`).
///
/// # Errors
///
/// [`GyreDataError::Unported`] for an attached atmosphere (`build_atm`);
/// [`GyreDataError::Upstream`] where upstream stops.
#[allow(clippy::too_many_lines)]
pub fn get_osc_data(
    s: &PulseDataInput<'_>,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), GyreDataError> {
    let b = &s.base;
    let nz = b.r.len();
    let (k_a, k_b) = set_segment_indices(b, add_center_point)?;
    let n_sg = k_a.len();
    if add_atmosphere {
        return Err(GyreDataError::Unported {
            what: "add_atmosphere (build_atm)",
            at: "star/private/pulse_osc.f90:126-136",
        });
    }
    let nn_atm = 0usize;
    let n_env = nz;
    let nn_env = if keep_surface_point {
        n_env + n_sg - 1
    } else {
        n_env - 1 + n_sg - 1
    };
    let nn = if add_center_point {
        nn_env + nn_atm + 1
    } else {
        nn_env + nn_atm
    };

    let mut g = vec![0.0_f64; OSC_ICONST];
    let r_outer = RSUN * b.photosphere_r;
    let m_outer = at(b.m_grav, 1);
    g[0] = m_outer;
    g[1] = r_outer;
    g[2] = at(b.l, 1);
    g[3] = s.initial_z;
    g[4] = 1.0 - (s.initial_y + s.initial_z);
    g[5] = s.mixing_length_alpha;
    if let Some(k) = s.largest_conv_mixing_region_bottom {
        g[6] = at(&s.x, k);
        g[7] = at(&s.y, k);
    }
    let rho_c = if b.interpolate_rho_for_pulse_data {
        eval_center_rho(b, k_b[n_sg - 1])
    } else {
        eval_center(b.rmid, b.rho, k_a[n_sg - 1], k_b[n_sg - 1])?
    };
    // at the centre d2P/dr2 = -4 pi G rho^2/3
    let d2p_dr2_c = -(FOUR_THIRDS * PI * at(b.cgrav, nz) * (rho_c * rho_c));
    let rm = at(b.rmid, nz);
    let p_c = at(b.peos, nz) - 0.5 * d2p_dr2_c * (rm * rm);
    g[8] = r_outer * r_outer * d2p_dr2_c / p_c;
    g[9] = r_outer
        * r_outer
        * eval_center_d2(b.rmid, b.rho, k_a[n_sg - 1], k_b[n_sg - 1])?
        / rho_c;
    g[10] = s.star_age;
    g[11] = if b.rotation_flag { at(b.omega, 1) } else { 0.0 };
    // global_data(13) (the initial rotation rate) stays 0 upstream.

    let mut pd = vec![vec![0.0_f64; OSC_IVAR + OSC_IABUND]; nn];
    let dq = b.dq;
    let iso = s.iso;

    let abundances =
        |p: &mut [f64],
         x: &dyn Fn(Option<usize>) -> Result<f64, GyreDataError>|
         -> Result<(), GyreDataError> {
            p[22] = x(iso.h1)?;
            p[23] = x(iso.h2)?;
            p[24] = x(iso.he3)?;
            p[25] = x(iso.he4)?;
            p[26] = x(iso.li7)?;
            p[27] = x(iso.be7)?;
            p[28] = x(iso.c12)?;
            p[29] = x(iso.c13)?;
            p[30] = x(iso.n14)?;
            p[31] = x(iso.n15)?;
            p[32] = x(iso.o16)?;
            p[33] = x(iso.o17)?;
            p[34] = x(iso.be9)?;
            p[35] = x(iso.si28)?;
            Ok(())
        };

    // store_point_data_env (pulse_osc.f90:340-441)
    let store_env = |p: &mut [f64],
                     k: usize,
                     k_a: usize,
                     k_b: usize|
     -> Result<(), GyreDataError> {
        p[0] = at(b.r, k);
        p[1] = flexmesa_math::log(at(b.m_grav, k) / m_outer);
        p[2] = eval_face(dq, b.t, k, 1, nz)?;
        p[3] = eval_face(dq, b.peos, k, 1, nz)?;
        p[4] = if b.interpolate_rho_for_pulse_data {
            eval_face(dq, b.rho, k, k_a, k_b)?
        } else {
            eval_face_rho(b, k, k_a, k_b)?
        };
        // "Not quite right; gradT can be discontinuous"
        p[5] = at(b.grad_t, k);
        p[6] = at(b.l, k);
        let kap = eval_face(dq, b.opacity, k, k_a, k_b)?;
        p[7] = kap;
        p[8] = eval_face(dq, b.eps_nuc, k, k_a, k_b)?
            + eval_face(dq, &s.eps_grav, k, k_a, k_b)?;
        p[9] = eval_face(dq, b.gamma1, k, k_a, k_b)?;
        p[10] = eval_face(dq, b.grada, k, k_a, k_b)?;
        p[11] = eval_face(dq, b.chi_t, k, k_a, k_b)?
            / eval_face(dq, b.chi_rho, k, k_a, k_b)?;
        p[12] = eval_face(dq, s.cp, k, k_a, k_b)?;
        p[13] = flexmesa_math::exp(eval_face(dq, s.lnfree_e, k, k_a, k_b)?);
        p[14] = eval_face_a_ast(b, k, k_a, k_b)?;
        p[15] = if b.rotation_flag { at(b.omega, k) } else { 0.0 };
        p[16] = eval_face(dq, b.d_opacity_dlnt, k, k_a, k_b)? / kap;
        p[17] = eval_face(dq, b.d_opacity_dlnd, k, k_a, k_b)? / kap;
        p[18] = eval_face(dq, b.d_epsnuc_dlnt, k, k_a, k_b)?;
        p[19] = eval_face(dq, b.d_epsnuc_dlnd, k, k_a, k_b)?;
        p[20] = eval_face(dq, b.peos, k, 1, nz)?
            / eval_face(dq, s.pgas, k, k_a, k_b)?;
        p[21] = at(s.gradr, k);
        abundances(p, &|i| s.eval_face_x(i, k, k_a, k_b))?;
        Ok(())
    };

    let mut j = 0usize;
    let mut sg = 0usize;
    for k in 1..=n_env {
        if k == 1 && !keep_surface_point {
            continue;
        }
        store_env(&mut pd[j], k, k_a[sg], k_b[sg])?;
        j += 1;
        if k == k_b[sg] + 1 {
            sg += 1;
            store_env(&mut pd[j], k, k_a[sg], k_b[sg])?;
            j += 1;
        }
    }

    // store_point_data_ctr (pulse_osc.f90:444-550)
    if add_center_point {
        let (k_a, k_b) = (k_a[n_sg - 1], k_b[n_sg - 1]);
        let rmid = b.rmid;
        let p = &mut pd[j];
        p[0] = 0.0;
        p[1] = flexmesa_math::log(f64::MIN_POSITIVE);
        p[2] = eval_center(rmid, b.t, 1, nz)?;
        p[3] = p_c;
        p[4] = rho_c;
        p[5] = eval_center(b.r, b.grad_t, k_a, k_b)?;
        p[6] = 0.0;
        let kap = eval_center(rmid, b.opacity, k_a, k_b)?;
        p[7] = kap;
        p[8] = eval_center(rmid, b.eps_nuc, k_a, k_b)?
            + eval_center(rmid, &s.eps_grav, k_a, k_b)?;
        p[9] = eval_center(rmid, b.gamma1, k_a, k_b)?;
        p[10] = eval_center(rmid, b.grada, k_a, k_b)?;
        p[11] = eval_center(rmid, b.chi_t, k_a, k_b)?
            / eval_center(rmid, b.chi_rho, k_a, k_b)?;
        p[12] = eval_center(rmid, s.cp, k_a, k_b)?;
        p[13] = flexmesa_math::exp(eval_center(rmid, s.lnfree_e, k_a, k_b)?);
        // `A_ast = point_data(15,j)`: the zero the array was filled with.
        p[15] = if b.rotation_flag {
            eval_center(b.r, b.omega, k_a, k_b)?
        } else {
            0.0
        };
        p[16] = eval_center(rmid, b.d_opacity_dlnt, k_a, k_b)? / kap;
        p[17] = eval_center(rmid, b.d_opacity_dlnd, k_a, k_b)? / kap;
        p[18] = eval_center(rmid, b.d_epsnuc_dlnt, k_a, k_b)?;
        p[19] = eval_center(rmid, b.d_epsnuc_dlnd, k_a, k_b)?;
        p[20] = eval_center(rmid, b.peos, 1, nz)?
            / eval_center(rmid, s.pgas, k_a, k_b)?;
        p[21] = eval_center(b.r, s.gradr, k_a, k_b)?;
        abundances(p, &|i| s.eval_center_x(i, k_a, k_b))?;
        j += 1;
    }
    if j != nn {
        return Err(GyreDataError::Upstream(
            "get_osc_data: point count mismatch",
        ));
    }
    Ok((g, pd))
}
