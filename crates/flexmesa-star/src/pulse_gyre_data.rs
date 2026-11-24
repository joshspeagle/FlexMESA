// SPDX-License-Identifier: GPL-3.0-or-later
//! In-memory GYRE pulsation data: `star_get_pulse_data(id, 'GYRE', ...)`,
//! i.e. `get_gyre_data` (`star/private/pulse_gyre.f90:38-416`), with the
//! `pulse_utils` helpers it calls (`star/private/pulse_utils.f90`) and
//! `interp_val_to_pt` (`star/private/star_utils.f90:779-811`), transcribed
//! in upstream order of operations (#170, package G3b).
//!
//! The producer reads a [`GyreDataInput`] view of the star (MESA cell
//! index `k = 1..nz`, surface first, stored 0-based) and returns
//! `global_data(3)` and `point_data(n_cols, nn)` in GYRE order (centre to
//! surface), which is what MESA passes to GYRE's `set_model`.
//!
//! Supported: schemas 101 and 110, with or without the central point,
//! with or without the surface point, interpolated or mass-conserving face
//! densities, and double points without a cap on their number. The paths
//! that need routines not ported here return
//! [`GyreDataError::Unported`]: schema 120 (`eval_eps_grav_and_partials`),
//! `add_atmosphere` (`build_atm`), and `max_number_of_double_points > 0`
//! (`qsort`).

use flexmesa_const::{FOUR_THIRDS, PI, PI4, RSUN};

use crate::model::StarModel;

/// Errors of [`get_gyre_data`].
#[derive(Debug, Clone, PartialEq)]
pub enum GyreDataError {
    /// An upstream `mesa_error` stop or `ierr /= 0` return.
    Upstream(&'static str),
    /// A branch that needs routines not ported yet.
    Unported {
        /// What the branch does.
        what: &'static str,
        /// Upstream location.
        at: &'static str,
    },
}

impl std::fmt::Display for GyreDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upstream(m) => write!(f, "get_gyre_data: {m}"),
            Self::Unported { what, at } => {
                write!(f, "get_gyre_data path not ported: {what} ({at})")
            }
        }
    }
}

impl std::error::Error for GyreDataError {}

/// The star state `get_gyre_data` reads, as MESA's `star_info` fields.
/// Every slice has length `nz`; element `k-1` is MESA cell or face `k`.
#[derive(Debug, Clone, Copy)]
pub struct GyreDataInput<'a> {
    /// `s%r` (cm, faces).
    pub r: &'a [f64],
    /// `s%m` (g, faces).
    pub m: &'a [f64],
    /// `s%m_grav` (g, faces).
    pub m_grav: &'a [f64],
    /// `s%dm` (g, cells).
    pub dm: &'a [f64],
    /// `s%dq` (cells).
    pub dq: &'a [f64],
    /// `s%rmid` (cm, cells).
    pub rmid: &'a [f64],
    /// `s%L` (erg/s, faces).
    pub l: &'a [f64],
    /// `s%Peos` (cells).
    pub peos: &'a [f64],
    /// `s%rho` (cells).
    pub rho: &'a [f64],
    /// `s%T` (cells).
    pub t: &'a [f64],
    /// `s%mu` (cells), used only for double points.
    pub mu: &'a [f64],
    /// `s%brunt_N2` (faces).
    pub brunt_n2: &'a [f64],
    /// `s%grav` (faces).
    pub grav: &'a [f64],
    /// `s%cgrav` (faces).
    pub cgrav: &'a [f64],
    /// `s%gamma1` (cells).
    pub gamma1: &'a [f64],
    /// `s%grada` (cells).
    pub grada: &'a [f64],
    /// `s%chiT` (cells).
    pub chi_t: &'a [f64],
    /// `s%chiRho` (cells).
    pub chi_rho: &'a [f64],
    /// `s%gradT` (faces).
    pub grad_t: &'a [f64],
    /// `s%opacity` (cells).
    pub opacity: &'a [f64],
    /// `s%d_opacity_dlnT` (cells).
    pub d_opacity_dlnt: &'a [f64],
    /// `s%d_opacity_dlnd` (cells).
    pub d_opacity_dlnd: &'a [f64],
    /// `s%eps_nuc` (cells).
    pub eps_nuc: &'a [f64],
    /// `s%d_epsnuc_dlnT` (cells).
    pub d_epsnuc_dlnt: &'a [f64],
    /// `s%d_epsnuc_dlnd` (cells).
    pub d_epsnuc_dlnd: &'a [f64],
    /// `s%omega` (faces).
    pub omega: &'a [f64],
    /// `s%photosphere_r` ($R_\odot$).
    pub photosphere_r: f64,
    /// `s%gyre_data_schema`.
    pub gyre_data_schema: i32,
    /// `s%rotation_flag`.
    pub rotation_flag: bool,
    /// `s%interpolate_rho_for_pulse_data`.
    pub interpolate_rho_for_pulse_data: bool,
    /// `s%add_double_points_to_pulse_data`.
    pub add_double_points_to_pulse_data: bool,
    /// `s%threshold_grad_mu_for_double_point`.
    pub threshold_grad_mu_for_double_point: f64,
    /// `s%max_number_of_double_points`.
    pub max_number_of_double_points: i32,
    /// `s%calculate_Brunt_N2`.
    pub calculate_brunt_n2: bool,
}

/// 1-based view of a slice.
#[derive(Clone, Copy)]
struct V<'a>(&'a [f64]);

impl V<'_> {
    fn at(self, k: usize) -> f64 {
        self.0[k - 1]
    }
}

/// `set_segment_indices(s, k_a, k_b, include_last_face)`
/// (`pulse_utils.f90:42-133`), 1-based segment bounds.
pub(crate) fn set_segment_indices(
    s: &GyreDataInput<'_>,
    include_last_face: bool,
) -> Result<(Vec<usize>, Vec<usize>), GyreDataError> {
    let nz = s.r.len();
    let mut mask = vec![false; nz + 1];
    if s.add_double_points_to_pulse_data {
        let (mu, peos) = (V(s.mu), V(s.peos));
        let mut grad_mu = vec![0.0_f64; nz + 1];
        grad_mu[1] = 0.0;
        for k in 2..nz {
            grad_mu[k] = flexmesa_math::log(mu.at(k) / mu.at(k - 1))
                / flexmesa_math::log(peos.at(k) / peos.at(k - 1));
        }
        // After `do k = 2, s%nz-1`, k = s%nz.
        let k = nz;
        grad_mu[k] = if include_last_face {
            flexmesa_math::log(mu.at(nz) / mu.at(nz - 1))
                / flexmesa_math::log(peos.at(nz) / peos.at(nz - 1))
        } else {
            0.0
        };
        for k in 1..=nz {
            mask[k] = grad_mu[k].abs() > s.threshold_grad_mu_for_double_point;
        }
        if s.max_number_of_double_points > 0 {
            return Err(GyreDataError::Unported {
                what: "max_number_of_double_points > 0 (num_lib qsort)",
                at: "star/private/pulse_utils.f90:82-92",
            });
        }
    }
    let mut k_a = vec![1usize];
    let mut k_b = Vec::new();
    for k in 1..=nz {
        if mask[k] {
            k_b.push(k - 1);
            k_a.push(k);
        }
    }
    k_b.push(nz);
    Ok((k_a, k_b))
}

/// `interp_val_to_pt(v, k, sz, dq, str)` (`star_utils.f90:779-811`) on the
/// 1-based sub-arrays `v`, `dq` of length `sz`.
fn interp_val_to_pt(
    v: V<'_>,
    k: usize,
    sz: usize,
    dq: V<'_>,
) -> Result<f64, GyreDataError> {
    if k == 1 {
        return Ok(v.at(k));
    }
    if k > 2 && k < sz {
        return flexmesa_interp1d::interp_4_to_1(
            0.5 * (dq.at(k - 2) + dq.at(k - 1)),
            0.5 * (dq.at(k - 1) + dq.at(k)),
            0.5 * (dq.at(k) + dq.at(k + 1)),
            0.5 * dq.at(k - 2) + dq.at(k - 1),
            v.at(k - 2),
            v.at(k - 1),
            v.at(k),
            v.at(k + 1),
        )
        .map_err(|_| GyreDataError::Upstream("interp_val_to_pt"));
    }
    Ok((v.at(k) * dq.at(k - 1) + v.at(k - 1) * dq.at(k))
        / (dq.at(k - 1) + dq.at(k)))
}

/// [`interp_val_to_pt`] on plain 1-based sub-arrays (for `eval_face_X`).
pub(crate) fn interp_val_to_pt_slices(
    v: &[f64],
    k: usize,
    sz: usize,
    dq: &[f64],
) -> Result<f64, GyreDataError> {
    interp_val_to_pt(V(v), k, sz, V(dq))
}

/// `eval_face(dq, v, k, k_a, k_b)` (`pulse_utils.f90:136-189`), no limits.
pub(crate) fn eval_face(
    dq: &[f64],
    v: &[f64],
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<f64, GyreDataError> {
    let (dq, v) = (V(dq), V(v));
    if k_b < k_a {
        return Err(GyreDataError::Upstream(
            "eval_face: invalid segment indices",
        ));
    }
    if k < k_a || k > k_b + 1 {
        return Err(GyreDataError::Upstream(
            "eval_face: out-of-bounds interpolation",
        ));
    }
    Ok(if k_b == k_a {
        v.at(k_a)
    } else if k == k_a {
        v.at(k_a)
            - dq.at(k_a) * (v.at(k_a + 1) - v.at(k_a))
                / (dq.at(k_a + 1) + dq.at(k_a))
    } else if k == k_a + 1 {
        v.at(k_a)
            + dq.at(k_a) * (v.at(k_a + 1) - v.at(k_a))
                / (dq.at(k_a + 1) + dq.at(k_a))
    } else if k == k_b {
        v.at(k_b)
            - dq.at(k_b) * (v.at(k_b) - v.at(k_b - 1))
                / (dq.at(k_b) + dq.at(k_b - 1))
    } else if k == k_b + 1 {
        v.at(k_b)
            + dq.at(k_b) * (v.at(k_b) - v.at(k_b - 1))
                / (dq.at(k_b) + dq.at(k_b - 1))
    } else {
        // interp_val_to_pt(v(k_a:k_b), k-k_a+1, k_b-k_a+1, dq(k_a:k_b), ...)
        interp_val_to_pt(
            V(&v.0[k_a - 1..k_b]),
            k - k_a + 1,
            k_b - k_a + 1,
            V(&dq.0[k_a - 1..k_b]),
        )?
    })
}

/// `eval_face_A_ast(s, k, k_a, k_b)` (`pulse_utils.f90:248-298`).
pub(crate) fn eval_face_a_ast(
    s: &GyreDataInput<'_>,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<f64, GyreDataError> {
    let (n2, r, grav, dq) = (V(s.brunt_n2), V(s.r), V(s.grav), V(s.dq));
    if k_b < k_a {
        return Err(GyreDataError::Upstream(
            "eval_face_A_ast: invalid segment indices",
        ));
    }
    if k < k_a || k > k_b + 1 {
        return Err(GyreDataError::Upstream(
            "eval_face_A_ast: out-of-bounds interpolation",
        ));
    }
    if !s.calculate_brunt_n2 {
        return Err(GyreDataError::Upstream(
            "eval_face_A_ast: must have calculate_Brunt_N2 = .true.",
        ));
    }
    Ok(if k_b == k_a {
        n2.at(k) * r.at(k) / grav.at(k)
    } else if k == k_a {
        let a1 = n2.at(k_a + 1) * r.at(k_a + 1) / grav.at(k_a + 1);
        let a2 = n2.at(k_a + 2) * r.at(k_a + 2) / grav.at(k_a + 2);
        a1 - dq.at(k_a) * (a2 - a1) / dq.at(k_a + 1)
    } else if k == k_b + 1 {
        let a1 = n2.at(k_b - 1) * r.at(k_b - 1) / grav.at(k_b);
        let a2 = n2.at(k_b) * r.at(k_b) / grav.at(k_b);
        a2 + dq.at(k_b) * (a2 - a1) / dq.at(k_b - 1)
    } else {
        n2.at(k) * r.at(k) / grav.at(k)
    })
}

/// `eval_face_rho(s, k, k_a, k_b)` (`pulse_utils.f90:301-351`).
pub(crate) fn eval_face_rho(
    s: &GyreDataInput<'_>,
    k: usize,
    k_a: usize,
    k_b: usize,
) -> Result<f64, GyreDataError> {
    let (rr, dm, rmid, rho) = (V(s.r), V(s.dm), V(s.rmid), V(s.rho));
    if k_b < k_a {
        return Err(GyreDataError::Upstream(
            "eval_face_rho: invalid segment indices",
        ));
    }
    if k < k_a || k > k_b + 1 {
        return Err(GyreDataError::Upstream(
            "eval_face_rho: out-of-bounds interpolation",
        ));
    }
    if k_b == k_a {
        return Ok(rho.at(k));
    }
    let (r, d_m, dlnr) = if k == k_a {
        let r = rr.at(k_a);
        (r, 0.5 * dm.at(k_a), 1.0 - rmid.at(k_a) / r)
    } else if k == k_b + 1 {
        let r = rr.at(k_b + 1);
        (r, 0.5 * dm.at(k_b), rmid.at(k_b) / r - 1.0)
    } else {
        let r = rr.at(k);
        (
            r,
            0.5 * (dm.at(k) + dm.at(k - 1)),
            (rmid.at(k - 1) - rmid.at(k)) / r,
        )
    };
    Ok(d_m / (PI4 * r * r * r * dlnr))
}

/// `eval_center(r, v, k_a, k_b)` (`pulse_utils.f90:354-404`), no limits.
pub(crate) fn eval_center(
    r: &[f64],
    v: &[f64],
    k_a: usize,
    k_b: usize,
) -> Result<f64, GyreDataError> {
    let (r, v) = (V(r), V(v));
    if k_b < k_a {
        return Err(GyreDataError::Upstream(
            "eval_center: invalid segment indices",
        ));
    }
    Ok(if k_a == k_b {
        v.at(k_a)
    } else {
        let r_1 = r.at(k_b);
        let r_2 = r.at(k_b - 1);
        let v_1 = v.at(k_b);
        let v_2 = v.at(k_b - 1);
        (v_1 * r_2 * r_2 - v_2 * r_1 * r_1) / (r_2 * r_2 - r_1 * r_1)
    })
}

/// `eval_center_rho(s, k_b)` (`pulse_utils.f90:463-482`).
pub(crate) fn eval_center_rho(s: &GyreDataInput<'_>, k_b: usize) -> f64 {
    let r_1 = V(s.rmid).at(k_b);
    let rho_1 = V(s.rho).at(k_b);
    let m_1 = V(s.m).at(k_b) - 0.5 * V(s.dm).at(k_b);
    3.0 * (5.0 * m_1 / (PI * r_1 * r_1 * r_1) - 4.0 * rho_1) / 8.0
}

/// Columns of the 1.01/1.10 point data (`pulse_gyre.f90:134-155`).
const N_COLS: usize = 18;

/// `get_gyre_data(id, add_center_point, keep_surface_point, add_atmosphere,
/// global_data, point_data, ierr)` (`pulse_gyre.f90:38-416`).
///
/// Returns `global_data = [M, R, L]` and the point data with
/// `point_data[j][c]` holding `point_data(c+1, j+1)` upstream, `j` from
/// the centre outwards.
///
/// # Errors
///
/// [`GyreDataError::Unported`] for schema 120, an attached atmosphere and
/// capped double points; [`GyreDataError::Upstream`] where upstream stops.
#[allow(clippy::too_many_lines)]
pub fn get_gyre_data(
    s: &GyreDataInput<'_>,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), GyreDataError> {
    let nz = s.r.len();
    let (k_a, k_b) = set_segment_indices(s, add_center_point)?;
    let n_sg = k_a.len();
    if add_atmosphere {
        return Err(GyreDataError::Unported {
            what: "add_atmosphere (build_atm)",
            at: "star/private/pulse_gyre.f90:99-110",
        });
    }
    let (n_atm, nn_atm) = (0usize, 0usize);
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
    match s.gyre_data_schema {
        101 | 110 => {}
        120 => {
            return Err(GyreDataError::Unported {
                what: "gyre_data_schema 120 (eps_grav column)",
                at: "star/private/pulse_gyre.f90:157-180, 191-201",
            })
        }
        _ => return Err(GyreDataError::Upstream("invalid gyre_data_schema")),
    }
    let global_data =
        vec![V(s.m_grav).at(1), RSUN * s.photosphere_r, V(s.l).at(1)];

    // point_data(18, nn) in storage order (surface first).
    let mut pd = vec![[0.0_f64; N_COLS]; nn + 1];
    let (rr, m_grav, ll, dq) = (V(s.r), V(s.m_grav), V(s.l), s.dq);
    let mut j = 1usize;
    let _ = n_atm;

    // store_point_data_env (pulse_gyre.f90:317-362)
    let store_env = |j: usize,
                     k: usize,
                     k_a: usize,
                     k_b: usize,
                     pd: &mut Vec<[f64; N_COLS]>|
     -> Result<(), GyreDataError> {
        let p = &mut pd[j];
        p[0] = rr.at(k);
        p[1] = m_grav.at(k);
        p[2] = ll.at(k);
        p[3] = eval_face(dq, s.peos, k, 1, nz)?;
        p[5] = if s.interpolate_rho_for_pulse_data {
            eval_face(dq, s.rho, k, k_a, k_b)?
        } else {
            eval_face_rho(s, k, k_a, k_b)?
        };
        p[4] = eval_face(dq, s.t, k, 1, nz)?;
        p[7] = eval_face_a_ast(s, k, k_a, k_b)? * V(s.grav).at(k) / rr.at(k);
        p[8] = eval_face(dq, s.gamma1, k, k_a, k_b)?;
        p[9] = eval_face(dq, s.grada, k, k_a, k_b)?;
        p[10] = eval_face(dq, s.chi_t, k, k_a, k_b)?
            / eval_face(dq, s.chi_rho, k, k_a, k_b)?;
        p[6] = V(s.grad_t).at(k);
        p[11] = eval_face(dq, s.opacity, k, k_a, k_b)?;
        p[12] = eval_face(dq, s.d_opacity_dlnt, k, k_a, k_b)?;
        p[13] = eval_face(dq, s.d_opacity_dlnd, k, k_a, k_b)?;
        p[14] = eval_face(dq, s.eps_nuc, k, k_a, k_b)?;
        p[15] = eval_face(dq, s.d_epsnuc_dlnt, k, k_a, k_b)?;
        p[16] = eval_face(dq, s.d_epsnuc_dlnd, k, k_a, k_b)?;
        if s.rotation_flag {
            p[17] = V(s.omega).at(k);
        } else {
            // Upstream assigns the whole column (`Omega_rot = 0d0`).
            for q in pd.iter_mut() {
                q[17] = 0.0;
            }
        }
        Ok(())
    };

    let mut sg = 0usize;
    for k in 1..=n_env {
        if k == 1 && !keep_surface_point {
            continue;
        }
        store_env(j, k, k_a[sg], k_b[sg], &mut pd)?;
        j += 1;
        if k == k_b[sg] + 1 {
            sg += 1;
            store_env(j, k, k_a[sg], k_b[sg], &mut pd)?;
            j += 1;
        }
    }

    // store_point_data_ctr (pulse_gyre.f90:365-414)
    if add_center_point {
        let (k_a, k_b) = (k_a[n_sg - 1], k_b[n_sg - 1]);
        let rmid = s.rmid;
        let p = &mut pd[j];
        p[0] = 0.0;
        p[1] = 0.0;
        p[2] = 0.0;
        p[3] = eval_center(rmid, s.peos, 1, nz)?;
        p[5] = if s.interpolate_rho_for_pulse_data {
            eval_center(rmid, s.rho, k_a, k_b)?
        } else {
            eval_center_rho(s, k_b)
        };
        // at the centre d2P/dr2 = -4 pi G rho^2/3
        let d2p_dr2_c = -(FOUR_THIRDS * PI * V(s.cgrav).at(nz) * (p[5] * p[5]));
        let rm = V(rmid).at(nz);
        p[3] = V(s.peos).at(nz) - 0.5 * d2p_dr2_c * (rm * rm);
        p[4] = eval_center(rmid, s.t, 1, nz)?;
        p[7] = 0.0;
        p[8] = eval_center(rmid, s.gamma1, k_a, k_b)?;
        p[9] = eval_center(rmid, s.grada, k_a, k_b)?;
        p[10] = eval_center(rmid, s.chi_t, k_a, k_b)?
            / eval_center(rmid, s.chi_rho, k_a, k_b)?;
        p[6] = eval_center(s.r, s.grad_t, k_a, k_b)?;
        p[11] = eval_center(rmid, s.opacity, k_a, k_b)?;
        p[12] = eval_center(rmid, s.d_opacity_dlnt, k_a, k_b)?;
        p[13] = eval_center(rmid, s.d_opacity_dlnd, k_a, k_b)?;
        p[14] = eval_center(rmid, s.eps_nuc, k_a, k_b)?;
        p[15] = eval_center(rmid, s.d_epsnuc_dlnt, k_a, k_b)?;
        p[16] = eval_center(rmid, s.d_epsnuc_dlnd, k_a, k_b)?;
        p[17] = if s.rotation_flag {
            eval_center(s.r, s.omega, k_a, k_b)?
        } else {
            0.0
        };
        j += 1;
    }

    if j != nn + 1 {
        return Err(GyreDataError::Upstream(
            "Invalid cell index in get_gyre_data",
        ));
    }
    // point_data = point_data(:,nn:1:-1)
    let point_data: Vec<Vec<f64>> =
        (1..=nn).rev().map(|jj| pd[jj].to_vec()).collect();
    Ok((global_data, point_data))
}

impl<'a> GyreDataInput<'a> {
    /// The view of a [`StarModel`] that `get_gyre_data` reads, with
    /// `photosphere_r` from [`crate::report_utils::eval_photosphere_info`],
    /// the routine that also supplies the `photosphere_r` history column.
    ///
    /// # Errors
    ///
    /// [`GyreDataError::Upstream`] if a model array is shorter than `nz`.
    pub fn from_model(model: &'a StarModel) -> Result<Self, GyreDataError> {
        let nz = model.nz;
        let take = |v: &'a [f64]| -> Result<&'a [f64], GyreDataError> {
            v.get(..nz)
                .ok_or(GyreDataError::Upstream("model array shorter than nz"))
        };
        let c = &model.controls;
        Ok(Self {
            r: take(&model.r)?,
            m: take(&model.m)?,
            m_grav: take(&model.m_grav)?,
            dm: take(&model.dm)?,
            dq: take(&model.dq)?,
            rmid: take(&model.rmid)?,
            l: take(&model.lum)?,
            peos: take(&model.peos)?,
            rho: take(&model.rho)?,
            t: take(&model.t)?,
            mu: take(&model.mu)?,
            brunt_n2: take(&model.brunt_n2)?,
            grav: take(&model.grav)?,
            cgrav: take(&model.cgrav)?,
            gamma1: take(&model.gamma1)?,
            grada: take(&model.grada)?,
            chi_t: take(&model.chi_t)?,
            chi_rho: take(&model.chi_rho)?,
            grad_t: take(&model.gradt)?,
            opacity: take(&model.opacity)?,
            d_opacity_dlnt: take(&model.d_opacity_dlnt)?,
            d_opacity_dlnd: take(&model.d_opacity_dlnd)?,
            eps_nuc: take(&model.eps_nuc)?,
            d_epsnuc_dlnt: take(&model.d_epsnuc_dlnt)?,
            d_epsnuc_dlnd: take(&model.d_epsnuc_dlnd)?,
            omega: take(&model.omega)?,
            photosphere_r: crate::report_utils::eval_photosphere_info(model)
                .r_rsun,
            gyre_data_schema: c.gyre_data_schema,
            rotation_flag: c.rotation_flag,
            interpolate_rho_for_pulse_data: c.interpolate_rho_for_pulse_data,
            add_double_points_to_pulse_data: c.add_double_points_to_pulse_data,
            threshold_grad_mu_for_double_point: c
                .threshold_grad_mu_for_double_point,
            max_number_of_double_points: c.max_number_of_double_points,
            calculate_brunt_n2: c.calculate_brunt_n2,
        })
    }
}

/// `star_get_pulse_data(id, 'GYRE', add_center_point, keep_surface_point,
/// add_atmosphere, global_data, point_data, ierr)`
/// (`get_pulse_data`, `star/private/pulse.f90:89-147`, `GYRE` branch) on a
/// FlexMESA model.
///
/// # Errors
///
/// As [`GyreDataInput::from_model`] and [`get_gyre_data`].
pub fn star_get_pulse_data_gyre(
    model: &StarModel,
    add_center_point: bool,
    keep_surface_point: bool,
    add_atmosphere: bool,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), GyreDataError> {
    let input = GyreDataInput::from_model(model)?;
    get_gyre_data(&input, add_center_point, keep_surface_point, add_atmosphere)
}
