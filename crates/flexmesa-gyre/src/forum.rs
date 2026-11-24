// SPDX-License-Identifier: GPL-3.0-only
//! ForUM 1.0.4 helpers that GYRE 8.1 calls (`forum/forum/src/lib/order_m.fypp`
//! in MESA 26.4.1): `sort_indices`, `unique_indices` and `locate` for
//! `real(RD)` arrays. ForUM's `reallocate` is plain resizing that keeps the
//! leading elements; ports use `Vec::resize`.
//!
//! Indices are Fortran-style (1-based) to keep the algorithms verbatim.

/// `sort_indices(x [, descend])`: Knuth's list merge sort (Algorithm L,
/// TAOCP Vol. 3), stable for ties, labels and variables as upstream.
///
/// Returns 1-based indices that sort `x` ascending (or the reverse when
/// `descend`).
///
/// # Panics
///
/// Panics if `x` has more than `i32::MAX - 2` elements.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn sort_indices(x: &[f64], descend: bool) -> Vec<i32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1];
    }
    let ni = i32::try_from(n).expect("array too long");
    // L(0:N+1)
    let mut l = vec![0_i32; n + 2];
    let at = |i: i32| usize::try_from(i).expect("negative index");
    let xv = |i: i32| x[at(i) - 1];
    // L1: prepare two lists
    l[0] = 1;
    for i in 1..=ni - 2 {
        l[at(i)] = -(i + 2);
    }
    l[n - 1] = 0;
    l[n] = 0;
    l[n + 1] = 2;
    // L2: begin new pass
    'l2: loop {
        let mut s: i32 = 0;
        let mut t: i32 = ni + 1;
        let mut p = l[at(s)];
        let mut q = l[at(t)];
        if q == 0 {
            break 'l2;
        }
        // L3: compare K_p : K_q
        loop {
            if xv(p) > xv(q) {
                // L6: advance q
                l[at(s)] = isign(q, l[at(s)]);
                s = q;
                q = l[at(q)];
                if q > 0 {
                    continue;
                }
                // L7: complete the sublist
                l[at(s)] = p;
                s = t;
                loop {
                    t = p;
                    p = l[at(p)];
                    if p <= 0 {
                        break;
                    }
                }
            } else {
                // L4: advance p
                l[at(s)] = isign(p, l[at(s)]);
                s = p;
                p = l[at(p)];
                if p > 0 {
                    continue;
                }
                // L5: complete the sublist
                l[at(s)] = q;
                s = t;
                loop {
                    t = q;
                    q = l[at(q)];
                    if q <= 0 {
                        break;
                    }
                }
            }
            // L8: end of pass?
            p = -p;
            q = -q;
            if q == 0 {
                l[at(s)] = isign(p, l[at(s)]);
                l[at(t)] = 0;
                break;
            }
        }
    }
    let mut idx = vec![0_i32; n];
    idx[0] = l[0];
    for i in 1..n {
        idx[i] = l[at(idx[i - 1])];
    }
    if descend {
        idx.reverse();
    }
    idx
}

/// Fortran integer `SIGN(a, b)`: `|a|` with the sign of `b` (`b = 0` counts
/// as positive).
#[inline]
fn isign(a: i32, b: i32) -> i32 {
    if b >= 0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// `unique_indices(x [, descend])`.
///
/// # Panics
///
/// Panics if `x` has more than `i32::MAX - 2` elements.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn unique_indices(x: &[f64], descend: bool) -> Vec<i32> {
    let mut idx = sort_indices(x, false);
    if idx.is_empty() {
        return idx;
    }
    let at = |i: i32| usize::try_from(i - 1).expect("bad index");
    let mut n = 1_usize;
    for i in 1..idx.len() {
        if x[at(idx[i])] != x[at(idx[n - 1])] {
            n += 1;
            idx[n - 1] = idx[i];
        }
    }
    idx.truncate(n);
    if descend {
        idx.reverse();
    }
    idx
}

/// `locate(x, x_loc, i_loc [, right])`: bracketing search in ascending `x`
/// starting from the guess `i_loc` (1-based, updated in place):
/// `x(i_loc) <= x_loc < x(i_loc+1)` (left) or
/// `x(i_loc-1) < x_loc <= x(i_loc)` (right); `0` below, `n+1` above.
///
/// # Panics
///
/// Panics if `x` is empty or has more than `i32::MAX - 1` elements.
pub fn locate(x: &[f64], x_loc: f64, i_loc: &mut i32, right: bool) {
    let n = i32::try_from(x.len()).expect("array too long");
    let xv = |i: i32| x[usize::try_from(i - 1).expect("bad index")];
    if x_loc < xv(1) {
        *i_loc = 0;
    } else if x_loc > xv(n) {
        *i_loc = n + 1;
    } else if right {
        locate_r(x, n, x_loc, i_loc);
    } else {
        locate_l(x, n, x_loc, i_loc);
    }
}

fn locate_l(x: &[f64], n: i32, x_loc: f64, i_loc: &mut i32) {
    let xv = |i: i32| x[usize::try_from(i - 1).expect("bad index")];
    let (mut i_lo, mut i_hi);
    if *i_loc < 1 || *i_loc > n {
        i_lo = 0;
        i_hi = n + 1;
    } else {
        let mut di = 1;
        if x_loc >= xv(*i_loc) {
            i_lo = *i_loc;
            loop {
                i_hi = i_lo + di;
                if i_hi > n {
                    i_hi = n + 1;
                    break;
                }
                if x_loc < xv(i_hi) {
                    break;
                }
                i_lo = i_hi;
                di *= 2;
            }
        } else {
            i_hi = *i_loc;
            loop {
                i_lo = i_hi - di;
                if i_lo < 1 {
                    i_lo = 0;
                    break;
                }
                if x_loc >= xv(i_lo) {
                    break;
                }
                i_hi = i_lo;
                di *= 2;
            }
        }
    }
    while i_hi - i_lo > 1 {
        let i_mid = (i_hi + i_lo) / 2;
        if x_loc >= xv(i_mid) {
            i_lo = i_mid;
        } else {
            i_hi = i_mid;
        }
    }
    *i_loc = i_lo;
}

fn locate_r(x: &[f64], n: i32, x_loc: f64, i_loc: &mut i32) {
    let xv = |i: i32| x[usize::try_from(i - 1).expect("bad index")];
    let (mut i_lo, mut i_hi);
    if *i_loc < 1 || *i_loc > n {
        i_lo = 0;
        i_hi = n + 1;
    } else {
        let mut di = 1;
        if x_loc <= xv(*i_loc) {
            i_hi = *i_loc;
            loop {
                i_lo = i_hi - di;
                if i_lo < 1 {
                    i_lo = 0;
                    break;
                }
                if x_loc > xv(i_lo) {
                    break;
                }
                i_hi = i_lo;
                di *= 2;
            }
        } else {
            i_lo = *i_loc;
            loop {
                i_hi = i_lo + di;
                if i_hi > n {
                    i_hi = n + 1;
                    break;
                }
                if x_loc <= xv(i_hi) {
                    break;
                }
                i_lo = i_hi;
                di *= 2;
            }
        }
    }
    while i_hi - i_lo > 1 {
        let i_mid = (i_hi + i_lo) / 2;
        if x_loc <= xv(i_mid) {
            i_hi = i_mid;
        } else {
            i_lo = i_mid;
        }
    }
    *i_loc = i_hi;
}
