// SPDX-License-Identifier: GPL-3.0-or-later
//! Upstream MESA `26.4.1` Lagrangian mesh split/merge point selection and
//! adjacent-ratio smoothing (`star/private/mesh_plan.f90:1-1301`).

use flexmesa_const::MixingType;

use crate::adjust_mesh_support::{
    fortran_maxloc_0based, fortran_minloc_0based,
};
use crate::controls::StarError;
use crate::controls_mix_wind::MixWindControls;
use crate::mesh_functions::MeshFunctionData;
use crate::model::StarModel;

/// Output of [`do_mesh_plan`] (`star/private/mesh_plan.f90:37-45`).
#[derive(Debug, Clone)]
pub struct MeshPlanOutput {
    /// New exterior mass-fraction coordinates `xq_new[0..nz_new]` (`xq_new[0] = 0`).
    pub xq_new: Vec<f64>,
    /// New cell mass fractions `dq_new[0..nz_new]`.
    pub dq_new: Vec<f64>,
    /// 1-based index of the mesh function that set `dq_new[k]` (`0` at center).
    pub which_gval: Vec<usize>,
    /// 0-based parent zone index `comes_from[k]` in `0..nz_old`.
    pub comes_from: Vec<usize>,
}

/// Linear root-finder matching `find0` in `num/public/num_lib.f90:517-534`.
#[must_use]
pub(crate) fn find0(xx1: f64, yy1: f64, xx2: f64, yy2: f64) -> f64 {
    let a = (xx1 * yy2) - (xx2 * yy1);
    let b = yy2 - yy1;
    let mut x = if b == 0.0
        || ((a.abs() >= b.abs() * 1.0e99)
            && ((yy1 >= 0.0 && yy2 <= 0.0) || (yy1 <= 0.0 && yy2 >= 0.0)))
    {
        0.5 * (xx1 + xx2)
    } else {
        a / b
    };
    if yy1 * yy2 <= 0.0 {
        let hi = xx1.max(xx2);
        let lo = xx1.min(xx2);
        if x > hi {
            x = hi;
        }
        if x < lo {
            x = lo;
        }
    }
    x
}

struct PlanState {
    nz_new: usize,
    xq_new: Vec<f64>,
    dq_new: Vec<f64>,
    which_gval: Vec<usize>,
    comes_from: Vec<usize>,
}

impl PlanState {
    fn with_capacity(nz_old: usize) -> Self {
        let cap = nz_old + 64;
        Self {
            nz_new: 0,
            xq_new: vec![0.0; cap],
            dq_new: vec![0.0; cap],
            which_gval: vec![0; cap],
            comes_from: vec![0; cap],
        }
    }

    fn ensure_1based(&mut self, idx1: usize) {
        if idx1 >= self.xq_new.len() {
            let new_len = ((self.xq_new.len() * 5) / 4 + 16).max(idx1 + 1);
            self.xq_new.resize(new_len, 0.0);
            self.dq_new.resize(new_len, 0.0);
            self.which_gval.resize(new_len, 0);
            self.comes_from.resize(new_len, 0);
        }
    }
}

struct PlanCtx<'a> {
    model: &'a StarModel,
    mw: &'a MixWindControls,
    xa: &'a [f64],
    nz_old: usize,
    max_allowed_nz: isize,
    okay_to_merge: bool,
    max_k_old_for_split: isize,
    min_k_old_for_split: isize,
    xq_old: Vec<f64>,
    dq_old: Vec<f64>,
    min_dq: f64,
    max_dq: f64,
    min_dq_for_split: f64,
    mesh_max_allowed_ratio: f64,
    do_not_split: Vec<bool>,
    num_gvals: usize,
    fcns: &'a MeshFunctionData,
    delta_gval_max: Vec<f64>,
    max_dq_cntr: f64,
    max_surface_cell_dq: f64,
    min_surface_cell_dq: f64,
    max_num_subcells: usize,
    max_num_merge_cells: usize,
    max_num_merge_surface_cells: usize,
}

impl PlanCtx<'_> {
    #[inline]
    fn gval(&self, k1: usize, j1: usize) -> f64 {
        self.fcns.vals[(j1 - 1) * self.nz_old + (k1 - 1)]
    }

    fn is_convective_boundary(&self, kk1: usize) -> bool {
        if kk1 >= self.nz_old {
            return false;
        }
        let m0 = self.model.mixing_type[kk1 - 1];
        let m1 = self.model.mixing_type[kk1];
        (m0 == MixingType::ConvectiveMixing
            && m1 != MixingType::ConvectiveMixing)
            || (m1 == MixingType::ConvectiveMixing
                && m0 != MixingType::ConvectiveMixing)
    }

    fn is_crystal_boundary(&self, kk1: usize) -> bool {
        if kk1 < 2 || !self.model.controls.do_phase_separation {
            return false;
        }
        let m_cryst = self.model.aux.crystal_core_boundary_mass;
        self.model.m[kk1 - 1] <= m_cryst && self.model.m[kk1 - 2] >= m_cryst
    }

    fn okay_to_split1(
        &self,
        k_old: usize,
        dq_new_k: f64,
        remaining_dq_old: f64,
    ) -> bool {
        if self.do_not_split[k_old] {
            return false;
        }
        if dq_new_k.max(remaining_dq_old)
            < self.min_dq_for_split.max(2.0 * self.min_dq)
        {
            return false;
        }
        if remaining_dq_old > 0.0 && dq_new_k > 0.99 * remaining_dq_old {
            return false;
        }
        if k_old < self.nz_old {
            let k0 = k_old - 1;
            let r_r = self.model.r[k0];
            let r_l = self.model.r[k0 + 1];
            let dr_old = r_r - r_l;
            let dq_parent = self.dq_old[k_old];

            let min_dr_cs = self.model.csound[k0] * self.mw.mesh_min_dr_div_cs;
            if (dr_old * dq_new_k) / dq_parent < 2.0 * min_dr_cs {
                return false;
            }

            let min_dr_star = self.mw.mesh_min_dr_div_drstar
                * (self.model.r[0] - self.model.r_center);
            if (dr_old * dq_new_k) / dq_parent < 2.0 * min_dr_star {
                return false;
            }

            if self.mw.mesh_min_dlnr > 0.0 {
                let dlnr_new = flexmesa_math::log(
                    (r_l + (dr_old * dq_new_k) / dq_parent) / r_l,
                );
                if dlnr_new < 2.0 * self.mw.mesh_min_dlnr {
                    return false;
                }
            }
        }
        true
    }

    fn split1(&self, st: &mut PlanState, k: usize) -> Result<(), StarError> {
        let k_old = st.comes_from[k];
        let mut k_old_last = 0_usize;
        let mut from_merger = st.xq_new[k] == self.xq_old[k_old]
            && st.dq_new[k] > self.dq_old[k_old] + self.min_dq * 1.0e-3;

        if from_merger {
            if k == st.nz_new {
                k_old_last = self.nz_old;
            } else {
                let xq_end = st.xq_new[k + 1];
                for kk in (k_old + 1)..=self.nz_old {
                    if self.xq_old[kk] == xq_end {
                        k_old_last = kk - 1;
                        break;
                    }
                    if self.xq_old[kk] > xq_end {
                        return Err(StarError::InvalidInput(format!(
                            "mesh_plan::split1: xq_old({kk})={} > xq_end={xq_end}",
                            self.xq_old[kk]
                        )));
                    }
                }
                if k_old_last == k_old {
                    from_merger = false;
                } else if k_old_last < k_old {
                    return Err(StarError::InvalidInput(format!(
                        "mesh_plan::split1: k_old_last={k_old_last} < k_old={k_old} at k={k}"
                    )));
                }
            }
        }

        let next_nz = st.nz_new + 1;
        if self.max_allowed_nz > 0 && (next_nz as isize) > self.max_allowed_nz {
            return Err(StarError::InvalidInput(format!(
                "mesh_plan tried to exceed max_allowed_nz={}",
                self.max_allowed_nz
            )));
        }
        st.ensure_1based(next_nz);
        st.nz_new = next_nz;

        for kk in ((k + 1)..=st.nz_new).rev() {
            st.xq_new[kk] = st.xq_new[kk - 1];
            st.dq_new[kk] = st.dq_new[kk - 1];
            st.which_gval[kk] = st.which_gval[kk - 1];
            st.comes_from[kk] = st.comes_from[kk - 1];
        }

        if from_merger {
            let xq_mid = st.xq_new[k] + st.dq_new[k] * 0.5;
            let mut split_at_k_old = k_old_last;
            for kk in (k_old + 1)..=k_old_last {
                if self.xq_old[kk] >= xq_mid {
                    if self.xq_old[kk] - xq_mid < xq_mid - self.xq_old[kk - 1] {
                        split_at_k_old = kk;
                    } else {
                        split_at_k_old = kk - 1;
                    }
                    break;
                }
            }
            st.comes_from[k + 1] = split_at_k_old;
            st.xq_new[k + 1] = self.xq_old[split_at_k_old];
            st.dq_new[k] = self.dq_old[k_old..split_at_k_old].iter().sum();
            st.dq_new[k + 1] =
                self.dq_old[split_at_k_old..=k_old_last].iter().sum();
        } else {
            let half = st.dq_new[k] * 0.5;
            st.dq_new[k] = half;
            st.dq_new[k + 1] = half;
            st.xq_new[k + 1] = st.xq_new[k] + st.dq_new[k];
            st.comes_from[k + 1] = self.nz_old;
            for ko in st.comes_from[k]..self.nz_old {
                if st.xq_new[k + 1] <= self.xq_old[ko + 1] {
                    st.comes_from[k + 1] = ko;
                    break;
                }
            }
        }
        Ok(())
    }

    fn round_off_xq(
        &self,
        k_old: usize,
        xq: f64,
    ) -> Result<(f64, f64), StarError> {
        let knxt = if xq >= self.xq_old[self.nz_old] {
            self.nz_old + 1
        } else {
            let mut j = 0_usize;
            for k in (1..=k_old).rev() {
                if self.xq_old[k] <= xq {
                    j = k + 1;
                    break;
                }
            }
            if j <= 1 {
                return Err(StarError::InvalidInput(String::from(
                    "logic error in mesh_plan::round_off_xq",
                )));
            }
            let mut found = self.nz_old + 1;
            for k in j..=self.nz_old {
                if self.xq_old[k] > xq {
                    found = k;
                    break;
                }
            }
            found
        };

        let mut dq = xq - self.xq_old[knxt - 1];
        let n = self.max_num_subcells;
        let sz = self.dq_old[knxt - 1] / (n as f64);
        let mut tmp = dq / sz;
        let huge_i32 = f64::from(i32::MAX);
        if tmp > huge_i32 {
            tmp = huge_i32;
        }
        let mut i = 1_i64.max(tmp.floor() as i64);
        if knxt > self.nz_old {
            i = i.min((n as i64) / 2);
        }
        dq = (i as f64) * sz;
        let rounded = self.xq_old[knxt - 1] + dq;
        if dq <= 0.0 || !dq.is_finite() {
            return Err(StarError::InvalidInput(format!(
                "mesh_plan::round_off_xq produced non-positive dq={dq}"
            )));
        }
        Ok((rounded, sz))
    }

    fn pick1_dq(
        &self,
        st: &PlanState,
        j1: usize,
        next_dq_max: f64,
        default: f64,
        k_old: usize,
        k_new: usize,
        min_dq_for_xa: f64,
        min_dq_for_logt: f64,
    ) -> Result<f64, StarError> {
        let is_xa = self.fcns.is_xa_function[j1 - 1];
        let is_logt = self.fcns.is_logt_function[j1 - 1];

        let dval = self.gval(k_old, j1) - self.gval(k_old + 1, j1);
        let gnew = self.gval(k_old + 1, j1)
            + dval * (self.xq_old[k_old + 1] - st.xq_new[k_new])
                / self.dq_old[k_old];
        let dgnew =
            self.delta_gval_max[k_old].min(self.delta_gval_max[k_old + 1]);
        let gmax = gnew + dgnew;
        let gmin = gnew - dgnew;

        let mut dq_sum = 0.0_f64;
        for k in (k_old + 1)..=self.nz_old {
            let gk = self.gval(k, j1);
            if gk <= gmax && gk >= gmin {
                if self.xq_old[k - 1] >= st.xq_new[k_new] {
                    dq_sum += self.dq_old[k - 1];
                } else {
                    if dq_sum != 0.0 {
                        return Err(StarError::InvalidInput(String::from(
                            "mesh_plan::pick1_dq: unexpected dq_sum != 0",
                        )));
                    }
                    dq_sum = self.xq_old[k] - st.xq_new[k_new];
                }
                if !dq_sum.is_finite() {
                    return Err(StarError::InvalidInput(format!(
                        "mesh_plan::pick1_dq: non-finite dq_sum={dq_sum}"
                    )));
                }
                if dq_sum >= next_dq_max {
                    break;
                }
                if dq_sum >= default {
                    dq_sum = default;
                    break;
                }
                if k < self.nz_old {
                    continue;
                }
                let gkm1 = self.gval(k - 1, j1);
                let dq_next = if gkm1 < gk {
                    find0(0.0, gkm1 - gmax, self.dq_old[k - 1], gk - gmax)
                } else if gkm1 > gk {
                    find0(0.0, gkm1 - gmin, self.dq_old[k - 1], gk - gmin)
                } else {
                    dq_sum = default;
                    break;
                };
                if dq_next > 1.0 - (st.xq_new[k_new] + self.min_dq) {
                    dq_sum = default;
                } else {
                    dq_sum += dq_next;
                }
                break;
            }

            let gkm1 = self.gval(k - 1, j1);
            let dq_next = if gk > gmax {
                find0(0.0, gkm1 - gmax, self.dq_old[k - 1], gk - gmax)
            } else {
                find0(0.0, gkm1 - gmin, self.dq_old[k - 1], gk - gmin)
            };
            if !dq_next.is_finite() {
                return Err(StarError::InvalidInput(format!(
                    "mesh_plan::pick1_dq: non-finite dq_next={dq_next}"
                )));
            }
            if self.xq_old[k - 1] >= st.xq_new[k_new] {
                dq_sum += dq_next;
            } else {
                if dq_sum != 0.0 {
                    return Err(StarError::InvalidInput(String::from(
                        "mesh_plan::pick1_dq: unexpected dq_sum != 0 on crossing",
                    )));
                }
                dq_sum = dq_next - (st.xq_new[k_new] - self.xq_old[k - 1]);
            }
            break;
        }

        dq_sum = self.min_dq.max(dq_sum);
        let xq_cand = st.xq_new[k_new] + dq_sum;
        let (xq, sz) = self.round_off_xq(k_old, xq_cand)?;
        let mut pick1 = self.min_dq.max(sz).max(xq - st.xq_new[k_new]);

        if is_xa && pick1 < min_dq_for_xa {
            pick1 = min_dq_for_xa;
        }
        if is_logt && pick1 < min_dq_for_logt {
            pick1 = min_dq_for_logt;
        }
        Ok(pick1)
    }

    fn pick_next_dq(
        &self,
        st: &mut PlanState,
        next_dq_max: f64,
        k_old: usize,
        k_new: usize,
        min_dq_for_xa: f64,
        min_dq_for_logt: f64,
    ) -> Result<f64, StarError> {
        let mut pick = if k_new == 1 {
            (1.0_f64).min(self.min_dq.sqrt())
        } else if k_new <= 20 {
            (1.0 - st.xq_new[k_new]).min(10.0 * st.dq_new[k_new - 1])
        } else {
            1.0 - st.xq_new[k_new]
        };

        if k_old == self.nz_old {
            st.which_gval[k_new] = 0;
            let rem = 1.0 - st.xq_new[k_new];
            pick = self.min_dq.max(rem - self.max_dq_cntr);
            for i in 1..=10 {
                if rem <= self.max_dq_cntr * (i as f64) {
                    pick = rem / (i as f64);
                    break;
                }
            }
            return Ok(pick);
        }

        let default = pick;
        let mut nxt_dqs = vec![default; self.num_gvals];
        for j1 in 1..=self.num_gvals {
            nxt_dqs[j1 - 1] = self.pick1_dq(
                st,
                j1,
                next_dq_max,
                default,
                k_old,
                k_new,
                min_dq_for_xa,
                min_dq_for_logt,
            )?;
        }

        let jmin0 = fortran_minloc_0based(&nxt_dqs);
        st.which_gval[k_new] = jmin0 + 1;
        Ok(self.min_dq.max(pick.min(nxt_dqs[jmin0])))
    }

    fn pick_new_points(&self, st: &mut PlanState) -> Result<(), StarError> {
        let mut k_old = 1_usize;
        let mut k_new = 1_usize;
        st.ensure_1based(1);
        st.xq_new[1] = 0.0;
        st.comes_from[1] = 1;
        let min_dr = self.mw.mesh_min_dr_div_drstar
            * (self.model.r[0] - self.model.r_center);
        let species = self.model.species;
        let has_v =
            self.model.controls.v_flag && self.model.v.len() >= self.nz_old;

        loop {
            let k_old_init = k_old;
            let next_dq_max = if k_new == 1 {
                self.max_surface_cell_dq
            } else {
                st.dq_new[k_new - 1] * self.mesh_max_allowed_ratio
            };

            let min_dq_for_xa = if self.model.gradr[k_old - 1]
                > self.model.grada[k_old - 1]
                && self.mw.min_dq_for_xa_convective > 0.0
            {
                self.mw.min_dq_for_xa_convective
            } else {
                self.mw.min_dq_for_xa
            };
            let min_dq_for_logt = self.mw.min_dq_for_logt;

            let mut next_dq = self.pick_next_dq(
                st,
                next_dq_max,
                k_old,
                k_new,
                min_dq_for_xa,
                min_dq_for_logt,
            )?;

            let max_merge = if k_new == 1 {
                self.max_num_merge_surface_cells
            } else {
                self.max_num_merge_cells
            };

            let remaining_dq_old = if k_old < self.nz_old {
                self.xq_old[k_old + 1] - st.xq_new[k_new]
            } else {
                1.0 - st.xq_new[k_new]
            };

            if next_dq < remaining_dq_old
                && !self.okay_to_split1(k_old, next_dq, remaining_dq_old)
            {
                next_dq = remaining_dq_old;
            }

            if next_dq > self.max_dq && st.xq_new[k_new] == self.xq_old[k_old] {
                for i in ((k_old + 1)..=self.nz_old).rev() {
                    if self.xq_old[i] - st.xq_new[k_new] <= self.max_dq {
                        next_dq = self.xq_old[i] - st.xq_new[k_new];
                        break;
                    }
                }
            }

            if next_dq > self.max_dq {
                next_dq =
                    self.xq_old[k_old] + self.dq_old[k_old] - st.xq_new[k_new];
            }

            if k_new == 1 && next_dq > self.max_surface_cell_dq {
                next_dq = self.max_surface_cell_dq;
            }

            if k_old == 1 && next_dq < self.min_surface_cell_dq {
                next_dq = self.min_surface_cell_dq;
            }

            let mut next_xq = st.xq_new[k_new] + next_dq;
            if next_xq > 1.0 - self.min_dq {
                next_xq = (1.0 + st.xq_new[k_new]) / 2.0;
                if k_old < self.nz_old && self.xq_old[k_old + 1] > next_xq {
                    next_xq = self.xq_old[k_old + 1];
                }
                next_dq = next_xq - st.xq_new[k_new];
            }

            if k_old < self.nz_old {
                let mut k_old_next;
                if st.xq_new[k_new] == self.xq_old[k_old]
                    && next_dq > self.dq_old[k_old] - self.min_dq / 2.0
                {
                    if !self.okay_to_merge
                        || (k_old as isize) < self.min_k_old_for_split
                        || (k_old as isize) > self.max_k_old_for_split
                    {
                        k_old_next = k_old + 1;
                    } else {
                        if next_dq > 1.5 * self.dq_old[k_old] {
                            next_dq *= 0.9;
                            next_xq = st.xq_new[k_new] + next_dq;
                        }
                        let k_old_next_max = self.nz_old.min(k_old + max_merge);
                        k_old_next = k_old_next_max;
                        for kk in (k_old + 1)..=k_old_next_max {
                            if self.mw.use_hydro_merge_limits_in_mesh_plan {
                                let mut du_div_cs_limit_flag = false;
                                if !self
                                    .mw
                                    .merge_amr_du_div_cs_limit_only_for_compression
                                {
                                    du_div_cs_limit_flag = true;
                                } else if has_v {
                                    if kk <= self.nz_old
                                        && self.model.v[kk - 1]
                                            * flexmesa_math::pow2(
                                                self.model.r[kk - 1],
                                            )
                                            > self.model.v[kk - 2]
                                                * flexmesa_math::pow2(
                                                    self.model.r[kk - 2],
                                                )
                                    {
                                        du_div_cs_limit_flag = true;
                                    }
                                    if !du_div_cs_limit_flag
                                        && kk - 1 > 1
                                        && self.model.v[kk - 2]
                                            * flexmesa_math::pow2(
                                                self.model.r[kk - 2],
                                            )
                                            > self.model.v[kk - 3]
                                                * flexmesa_math::pow2(
                                                    self.model.r[kk - 3],
                                                )
                                    {
                                        du_div_cs_limit_flag = true;
                                    }
                                }

                                let abs_du_div_cs = if du_div_cs_limit_flag
                                    && has_v
                                {
                                    if kk - 1 == 1 {
                                        (self.model.v[0] - self.model.v[1])
                                            .abs()
                                            / self.model.csound[0]
                                    } else if kk == self.nz_old {
                                        (self.model.v[self.nz_old - 2]
                                            - self.model.v[self.nz_old - 1])
                                            .abs()
                                            / self.model.csound[self.nz_old - 1]
                                    } else {
                                        (self.model.v[kk - 2]
                                            - self.model.v[kk - 1])
                                            .abs()
                                            / self.model.csound[kk - 2]
                                    }
                                } else {
                                    0.0
                                };

                                if du_div_cs_limit_flag
                                    && abs_du_div_cs
                                        > self.mw.merge_amr_max_abs_du_div_cs
                                    && (!self.mw.merge_amr_inhibit_at_jumps
                                        || self.dq_old[k_old] >= self.min_dq)
                                {
                                    k_old_next = kk - 1;
                                    break;
                                }
                            }

                            let xa_kk =
                                &self.xa[(kk - 1) * species..kk * species];
                            let xa_kkm1 = &self.xa
                                [(kk - 2) * species..(kk - 1) * species];
                            let maxval_delta_xa = xa_kk
                                .iter()
                                .zip(xa_kkm1)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0_f64, f64::max);
                            let j00 = fortran_maxloc_0based(xa_kk);
                            let jm1 = fortran_maxloc_0based(xa_kkm1);
                            if maxval_delta_xa > self.mw.max_delta_x_for_merge
                                || j00 != jm1
                                || self.is_convective_boundary(kk)
                                || self.is_crystal_boundary(kk)
                            {
                                k_old_next = kk - 1;
                                break;
                            } else if next_xq
                                <= self.xq_old[kk] + self.min_dq / 2.0
                            {
                                k_old_next = (k_old + 1).max(kk - 1);
                                break;
                            }
                        }
                    }

                    k_old_next = k_old_next.max(k_old + 1);
                    next_xq = self.xq_old[k_old_next];
                    next_dq = next_xq - st.xq_new[k_new];
                } else if next_xq >= self.xq_old[k_old + 1] - self.min_dq / 2.0
                {
                    k_old_next = k_old + 1;
                    next_xq = self.xq_old[k_old_next];
                    next_dq = next_xq - st.xq_new[k_new];
                } else {
                    k_old_next = k_old;
                }

                if k_old_next <= self.nz_old
                    && next_xq == self.xq_old[k_old_next]
                {
                    let mut force_merge_with_one_more = false;
                    if self.dq_old[k_old] < self.min_dq {
                        force_merge_with_one_more = true;
                    } else if self.mw.merge_if_dlnr_too_small {
                        if st.xq_new[k_new] <= self.xq_old[k_old]
                            && self.model.ln_r[k_old - 1]
                                - self.model.ln_r[k_old_next - 1]
                                < self.mw.mesh_min_dlnr
                        {
                            force_merge_with_one_more = true;
                        } else if k_old_next == self.nz_old
                            && self.model.r_center > 0.0
                        {
                            force_merge_with_one_more = self.model.ln_r
                                [k_old_next - 1]
                                - flexmesa_math::log(self.model.r_center)
                                < self.mw.mesh_min_dlnr;
                        }
                    }

                    if !force_merge_with_one_more
                        && self.mw.merge_if_dr_div_cs_too_small
                    {
                        if st.xq_new[k_new] <= self.xq_old[k_old]
                            && self.model.r[k_old - 1]
                                - self.model.r[k_old_next - 1]
                                < self.mw.mesh_min_dr_div_cs
                                    * self.model.csound[k_old - 1]
                        {
                            force_merge_with_one_more = true;
                        } else if k_old_next == self.nz_old {
                            force_merge_with_one_more = (self.model.r
                                [k_old_next - 1]
                                - self.model.r_center)
                                < self.mw.mesh_min_dr_div_cs
                                    * self.model.csound[k_old_next - 1];
                        }
                    }

                    if !force_merge_with_one_more
                        && self.mw.merge_if_dr_div_drstar_too_small
                    {
                        if st.xq_new[k_new] <= self.xq_old[k_old]
                            && self.model.r[k_old - 1]
                                - self.model.r[k_old_next - 1]
                                < min_dr
                        {
                            force_merge_with_one_more = true;
                        } else if k_old_next == self.nz_old {
                            force_merge_with_one_more = (self.model.r
                                [k_old_next - 1]
                                - self.model.r_center)
                                < min_dr;
                        }
                    }

                    if force_merge_with_one_more {
                        k_old_next += 1;
                        next_xq = if k_old_next < self.nz_old {
                            self.xq_old[k_old_next]
                        } else {
                            1.0
                        };
                        next_dq = next_xq - st.xq_new[k_new];
                    }
                }

                k_old = k_old_next.min(self.nz_old);
            }

            st.comes_from[k_new] = k_old_init;

            if 1.0 - st.xq_new[k_new] < self.max_dq_cntr
                || 1.0 - next_xq < self.min_dq
            {
                st.dq_new[k_new] = 1.0 - st.xq_new[k_new];
                break;
            }

            st.dq_new[k_new] = next_dq;

            if next_xq < st.xq_new[k_new] {
                return Err(StarError::InvalidInput(format!(
                    "mesh_plan::pick_new_points: next_xq={next_xq} < xq_new({k_new})={}",
                    st.xq_new[k_new]
                )));
            }

            let dq_sum: f64 = st.dq_new[1..=k_new].iter().sum();
            k_new += 1;
            if self.max_allowed_nz > 0 && (k_new as isize) > self.max_allowed_nz
            {
                return Err(StarError::InvalidInput(format!(
                    "mesh_plan tried to exceed max_allowed_nz={}",
                    self.max_allowed_nz
                )));
            }
            st.ensure_1based(k_new);
            st.xq_new[k_new] = next_xq;
            if (st.xq_new[k_new] - dq_sum).abs() > 1.0e-6 {
                return Err(StarError::InvalidInput(format!(
                    "mesh_plan::pick_new_points: |xq_new({k_new}) - dq_sum| = {} > 1e-6",
                    (st.xq_new[k_new] - dq_sum).abs()
                )));
            }

            while k_old < self.nz_old {
                if self.xq_old[k_old + 1] > next_xq {
                    break;
                }
                k_old += 1;
            }
        }

        st.nz_new = k_new;
        Ok(())
    }

    fn smooth_new_points(&self, st: &mut PlanState) -> Result<(), StarError> {
        let alfa = self.mesh_max_allowed_ratio;
        loop {
            if self.min_k_old_for_split <= 1
                && st.nz_new >= 2
                && st.dq_new[1] > alfa * st.dq_new[2]
                && self.okay_to_split1(1, st.dq_new[1], 0.0)
            {
                self.split1(st, 1)?;
                continue;
            }

            let mut done = true;
            let mut k = 2_usize;
            while k < st.nz_new {
                let k_old = st.comes_from[k];
                if (k_old as isize) <= self.max_k_old_for_split
                    && (k_old as isize) >= self.min_k_old_for_split
                    && self.okay_to_split1(k_old, st.dq_new[k], 0.0)
                    && (st.dq_new[k] > self.max_dq
                        || st.dq_new[k] > alfa * st.dq_new[k + 1]
                        || st.dq_new[k] > alfa * st.dq_new[k - 1])
                {
                    self.split1(st, k)?;
                    done = false;
                } else {
                    k += 1;
                }
            }

            if done {
                break;
            }

            done = true;
            let mut k = st.nz_new.saturating_sub(1);
            while k > 1 {
                if self.okay_to_split1(st.comes_from[k], st.dq_new[k], 0.0)
                    && (st.dq_new[k] > alfa * st.dq_new[k + 1]
                        || st.dq_new[k] > alfa * st.dq_new[k - 1])
                {
                    self.split1(st, k)?;
                    done = false;
                    k += 1;
                } else {
                    k -= 1;
                }
            }

            if done {
                break;
            }
        }
        Ok(())
    }
}

/// Runs upstream MESA `26.4.1` `do_mesh_plan` (`star/private/mesh_plan.f90:37-178`).
///
/// # Errors
/// Returns [`StarError::InvalidInput`] if mesh bounds are inconsistent, `xq_old`
/// is not strictly increasing, or `max_allowed_nz` is exceeded.
pub fn do_mesh_plan(
    model: &StarModel,
    mw: &MixWindControls,
    xq_old_0: &[f64],
    dq_old_0: &[f64],
    min_dq: f64,
    max_dq: f64,
    min_dq_for_split: f64,
    mesh_max_allowed_ratio: f64,
    do_not_split_0: &[bool],
    fcns: &MeshFunctionData,
    delta_gval_max_0: &[f64],
    max_center_cell_dq: f64,
    max_surface_cell_dq: f64,
    min_surface_cell_dq: f64,
) -> Result<MeshPlanOutput, StarError> {
    do_mesh_plan_with_xa(
        model,
        mw,
        &model.xa,
        xq_old_0,
        dq_old_0,
        min_dq,
        max_dq,
        min_dq_for_split,
        mesh_max_allowed_ratio,
        do_not_split_0,
        fcns,
        delta_gval_max_0,
        max_center_cell_dq,
        max_surface_cell_dq,
        min_surface_cell_dq,
    )
}

pub(crate) fn do_mesh_plan_with_xa<'a>(
    model: &'a StarModel,
    mw: &'a MixWindControls,
    xa_slice: &'a [f64],
    xq_old_0: &[f64],
    dq_old_0: &[f64],
    min_dq: f64,
    max_dq: f64,
    min_dq_for_split: f64,
    mesh_max_allowed_ratio: f64,
    do_not_split_0: &[bool],
    fcns: &'a MeshFunctionData,
    delta_gval_max_0: &[f64],
    max_center_cell_dq: f64,
    max_surface_cell_dq: f64,
    min_surface_cell_dq: f64,
) -> Result<MeshPlanOutput, StarError> {
    let nz_old = dq_old_0.len();
    if nz_old < 2
        || xq_old_0.len() != nz_old
        || delta_gval_max_0.len() != nz_old
    {
        return Err(StarError::InvalidInput(format!(
            "do_mesh_plan invalid slice lengths for nz_old={nz_old}"
        )));
    }
    let num_gvals = fcns.names.len();
    if num_gvals == 0 {
        return Err(StarError::InvalidInput(String::from(
            "must have at least 1 mesh gradient function with nonzero weight",
        )));
    }

    let max_k_old_for_split = if mw.mesh_max_k_old_for_split < 0 {
        (nz_old as isize) + mw.mesh_max_k_old_for_split
    } else {
        mw.mesh_max_k_old_for_split
    };
    let min_k_old_for_split = if mw.mesh_min_k_old_for_split < 0 {
        (nz_old as isize) + mw.mesh_min_k_old_for_split
    } else {
        mw.mesh_min_k_old_for_split
    };

    if max_dq < min_dq {
        return Err(StarError::InvalidInput(format!(
            "ERROR in controls: max_dq ({max_dq}) < min_dq ({min_dq})"
        )));
    }
    if max_center_cell_dq < min_dq {
        return Err(StarError::InvalidInput(format!(
            "ERROR in controls: max_center_cell_dq ({max_center_cell_dq}) < min_dq ({min_dq})"
        )));
    }
    if max_surface_cell_dq < min_dq {
        return Err(StarError::InvalidInput(format!(
            "ERROR in controls: max_surface_cell_dq ({max_surface_cell_dq}) < min_dq ({min_dq})"
        )));
    }

    for k in 1..nz_old {
        if xq_old_0[k] <= xq_old_0[k - 1] {
            return Err(StarError::InvalidInput(format!(
                "do_mesh_plan bad xq_old at k={}: {} <= {}",
                k + 1,
                xq_old_0[k],
                xq_old_0[k - 1]
            )));
        }
    }

    let mut xq_old = vec![0.0; nz_old + 1];
    xq_old[1..=nz_old].copy_from_slice(xq_old_0);
    let mut dq_old = vec![0.0; nz_old + 1];
    dq_old[1..=nz_old].copy_from_slice(dq_old_0);
    let mut do_not_split = vec![false; nz_old + 1];
    do_not_split[1..=nz_old].copy_from_slice(do_not_split_0);
    let mut delta_gval_max = vec![1.0; nz_old + 1];
    delta_gval_max[1..=nz_old].copy_from_slice(delta_gval_max_0);

    let max_dq_cntr = (1.0e-20_f64)
        .max(max_center_cell_dq)
        .max(0.75 * dq_old[nz_old]);

    let ctx = PlanCtx {
        model,
        mw,
        xa: xa_slice,
        nz_old,
        max_allowed_nz: mw.max_allowed_nz,
        okay_to_merge: mw.mesh_ok_to_merge,
        max_k_old_for_split,
        min_k_old_for_split,
        xq_old,
        dq_old,
        min_dq,
        max_dq,
        min_dq_for_split,
        mesh_max_allowed_ratio,
        do_not_split,
        num_gvals,
        fcns,
        delta_gval_max,
        max_dq_cntr,
        max_surface_cell_dq,
        min_surface_cell_dq,
        max_num_subcells: mw.max_num_subcells.max(1),
        max_num_merge_cells: mw.max_num_merge_cells,
        max_num_merge_surface_cells: mw.max_num_merge_surface_cells,
    };

    let mut st = PlanState::with_capacity(nz_old);
    ctx.pick_new_points(&mut st)?;

    if ctx.min_k_old_for_split <= 1 {
        let surf_cap = ctx
            .max_surface_cell_dq
            .max(2.0 * ctx.min_dq)
            .max(ctx.min_dq_for_split);
        while st.dq_new[1] > surf_cap {
            ctx.split1(&mut st, 1)?;
        }
    }

    ctx.smooth_new_points(&mut st)?;

    let nz_new = st.nz_new;
    Ok(MeshPlanOutput {
        xq_new: st.xq_new[1..=nz_new].to_vec(),
        dq_new: st.dq_new[1..=nz_new].to_vec(),
        which_gval: st.which_gval[1..=nz_new].to_vec(),
        comes_from: st.comes_from[1..=nz_new]
            .iter()
            .map(|&c1| c1 - 1)
            .collect(),
    })
}
