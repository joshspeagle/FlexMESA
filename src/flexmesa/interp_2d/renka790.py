# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scattered 2D cubic Shepard interpolation (ACM TOMS Algorithm 790).

Direct double-precision (`float64`) port of Robert J. Renka's `CSHEP2D`
package (`renka790_db.f` in upstream MESA `interp_2d/private/renka790_db.f`).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

_DTOL: float = 0.01
_RTOL: float = 1.0e-5


def _store2(
    x: np.ndarray,
    y: np.ndarray,
    nr: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Bin scattered nodes `(x, y)` into an `nr x nr` uniform spatial grid."""
    n = int(x.shape[0])
    if n < 2 or nr < 1:
        msg = f"store2 requires n >= 2 and nr >= 1, got n={n}, nr={nr}"
        raise ValueError(msg)
    xmn = float(np.min(x))
    xmx = float(np.max(x))
    ymn = float(np.min(y))
    ymx = float(np.max(y))
    delx = (xmx - xmn) / float(nr)
    dely = (ymx - ymn) / float(nr)
    if delx == 0.0 or dely == 0.0:
        msg = "collinear or duplicate scattered nodes in store2"
        raise ValueError(msg)

    lcell = np.zeros((nr, nr), dtype=np.int64)
    lnext = np.zeros(n, dtype=np.int64)
    for k in range(n, 0, -1):
        i = min(int((float(x[k - 1]) - xmn) / delx) + 1, nr)
        j = min(int((float(y[k - 1]) - ymn) / dely) + 1, nr)
        idx = int(lcell[i - 1, j - 1])
        lnext[k - 1] = idx if idx != 0 else k
        lcell[i - 1, j - 1] = k
    return lcell, lnext, xmn, ymn, delx, dely


def _getnp2(
    px: float,
    py: float,
    x: np.ndarray,
    y: np.ndarray,
    nr: int,
    lcell: np.ndarray,
    lnext: np.ndarray,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
) -> tuple[int, float]:
    """Locate the closest unmarked node to `(px, py)` and mark `lnext`."""
    if nr < 1 or dx <= 0.0 or dy <= 0.0:
        msg = "invalid cell grid parameters in getnp2"
        raise ValueError(msg)
    first = True
    imin, imax = 1, nr
    jmin, jmax = 1, nr
    delx = px - xmin
    dely = py - ymin
    i0 = min(max(int(delx / dx) + 1, 1), nr)
    j0 = min(max(int(dely / dy) + 1, 1), nr)
    i1 = i2 = i0
    j1 = j2 = j0
    lmin = 0
    rsmin = 0.0

    while True:
        for j in range(j1, j2 + 1):
            if j > jmax:
                break
            if j < jmin:
                continue
            for i in range(i1, i2 + 1):
                if i > imax:
                    break
                if i < imin:
                    continue
                if j not in (j1, j2) and i not in (i1, i2):
                    continue
                idx = int(lcell[i - 1, j - 1])
                if idx == 0:
                    continue
                while True:
                    ln = int(lnext[idx - 1])
                    if ln >= 0:
                        rsq = (float(x[idx - 1]) - px) ** 2 + (
                            float(y[idx - 1]) - py
                        ) ** 2
                        if first:
                            lmin = idx
                            rsmin = rsq
                            r = math.sqrt(rsmin)
                            imin = max(int((delx - r) / dx) + 1, 1)
                            imax = min(int((delx + r) / dx) + 1, nr)
                            jmin = max(int((dely - r) / dy) + 1, 1)
                            jmax = min(int((dely + r) / dy) + 1, nr)
                            first = False
                        elif rsq < rsmin:
                            lmin = idx
                            rsmin = rsq
                    next_l = abs(ln)
                    if next_l == idx:
                        break
                    idx = next_l
        if i1 <= imin and i2 >= imax and j1 <= jmin and j2 >= jmax:
            break
        i1 -= 1
        i2 += 1
        j1 -= 1
        j2 += 1

    if first:
        msg = "no unmarked node remaining in getnp2"
        raise ValueError(msg)
    lnext[lmin - 1] = -lnext[lmin - 1]
    return lmin, rsmin


def _setup2(
    xk: float,
    yk: float,
    zk: float,
    xi: float,
    yi: float,
    zi: float,
    s1: float,
    s2: float,
    s3: float,
    r: float,
) -> np.ndarray:
    """Build row `i` of weighted cubic least-squares matrix (`SETUP2_db`)."""
    dx = xi - xk
    dy = yi - yk
    dxsq = dx * dx
    dysq = dy * dy
    d = math.sqrt(dxsq + dysq)
    if d <= 0.0 or d >= r:
        return np.zeros(10, dtype=np.float64)
    w = (r - d) / r / d
    w1, w2, w3 = s1 * w, s2 * w, s3 * w
    return np.array(
        [
            dxsq * dx * w3,
            dxsq * dy * w3,
            dx * dysq * w3,
            dysq * dy * w3,
            dxsq * w2,
            dx * dy * w2,
            dysq * w2,
            dx * w1,
            dy * w1,
            (zi - zk) * w,
        ],
        dtype=np.float64,
    )


def _givens(a: float, b: float) -> tuple[float, float, float, float]:
    """Construct Givens plane rotation `(a_new, b_new, c, s)` (`GIVENS_db`)."""
    aa, bb = a, b
    if abs(aa) > abs(bb):
        u = aa + aa
        v = bb / u
        r = math.sqrt(0.25 + v * v) * u
        c = aa / r
        s = v * (c + c)
        return r, s, c, s
    if bb != 0.0:
        u = bb + bb
        v = aa / u
        r = math.sqrt(0.25 + v * v) * u
        s = bb / r
        c = v * (s + s)
        b_out = 1.0 / c if c != 0.0 else 1.0
        return r, b_out, c, s
    return aa, bb, 1.0, 0.0


@dataclasses.dataclass(frozen=True)
class Cshep2Interpolant:
    """Scattered 2D cubic Shepard interpolant (ACM TOMS Algorithm 790)."""

    x: np.ndarray
    y: np.ndarray
    f: np.ndarray
    nr: int
    lcell: np.ndarray
    lnext: np.ndarray
    xmin: float
    ymin: float
    dx: float
    dy: float
    rmax: float
    rw: np.ndarray
    a: np.ndarray

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        f: np.ndarray,
        nc: int = 17,
        nw: int = 32,
        nr: int = 6,
    ) -> Cshep2Interpolant:
        """Fit a scattered 2D cubic Shepard interpolant (`interp_cshep2_db`)."""
        xv = np.asarray(x, dtype=np.float64).ravel()
        yv = np.asarray(y, dtype=np.float64).ravel()
        fv = np.asarray(f, dtype=np.float64).ravel()
        n = int(xv.shape[0])
        lmax = min(40, n - 1)
        ncwmax = max(nc, nw)
        if (
            n < 10
            or yv.shape[0] != n
            or fv.shape[0] != n
            or nc < 9
            or nw < 1
            or ncwmax > lmax
            or nr < 1
        ):
            msg = f"invalid cshep2 parameters n={n}, nc={nc}, nw={nw}, nr={nr}"
            raise ValueError(msg)

        lcell, lnext, xmin, ymin, dx, dy = _store2(xv, yv, nr)
        npts = np.zeros(lmax, dtype=np.int64)
        rw = np.zeros(n, dtype=np.float64)
        a = np.zeros((n, 9), dtype=np.float64)
        rsmx = 0.0

        for k in range(1, n + 1):
            xk, yk, fk = float(xv[k - 1]), float(yv[k - 1]), float(fv[k - 1])
            lnext[k - 1] = -lnext[k - 1]

            rs = sum_rs = rws = rc = sfs = 0.0
            lnp = neq = 0

            while True:
                sum_rs += rs
                if lnp == lmax:
                    if rws == 0.0:
                        rws = 1.1 * rs
                    if rc == 0.0:
                        neq = lmax
                        rc = math.sqrt(1.1 * rs)
                        sfs = float(neq) / sum_rs
                    break
                lnp += 1
                rsold = rs
                np_idx, rs = _getnp2(
                    xk, yk, xv, yv, nr, lcell, lnext, xmin, ymin, dx, dy
                )
                if rs == 0.0:
                    msg = "duplicate scattered nodes in cshep2"
                    raise ValueError(msg)
                npts[lnp - 1] = np_idx
                if (rs - rsold) / rs < _RTOL:
                    continue
                if rws == 0.0 and lnp > nw:
                    rws = rs
                if rc == 0.0 and lnp > nc:
                    neq = lnp - 1
                    rc = math.sqrt(rs)
                    sfs = float(neq) / sum_rs
                if lnp > ncwmax:
                    break

            rw[k - 1] = math.sqrt(rws)
            rsmx = max(rsmx, rws)
            sf = math.sqrt(sfs)
            sfc = sf * sfs

            b = np.zeros((10, 10), dtype=np.float64)
            i_eq = 0

            while True:
                while i_eq < neq:
                    i_eq += 1
                    np_idx = int(npts[i_eq - 1])
                    irow = min(i_eq, 10) - 1
                    b[irow, :] = _setup2(
                        xk,
                        yk,
                        fk,
                        float(xv[np_idx - 1]),
                        float(yv[np_idx - 1]),
                        float(fv[np_idx - 1]),
                        sf,
                        sfs,
                        sfc,
                        rc,
                    )
                    if i_eq == 1:
                        continue
                    for j in range(irow):
                        b_jj, b_ij, c, s = _givens(
                            float(b[j, j]), float(b[irow, j])
                        )
                        b[j, j] = b_jj
                        b[irow, j] = b_ij
                        if not (c == 1.0 and s == 0.0):
                            row_j = b[j, j + 1 : 10].copy()
                            row_i = b[irow, j + 1 : 10].copy()
                            b[j, j + 1 : 10] = c * row_j + s * row_i
                            b[irow, j + 1 : 10] = -s * row_j + c * row_i

                dmin = min(abs(float(b[d_idx, d_idx])) for d_idx in range(9))
                if dmin * rc >= _DTOL:
                    break
                if neq == lmax:
                    stf = 1.0 / rc
                    for i_d in range(4):
                        b[9, :] = 0.0
                        b[9, i_d] = stf
                        for j in range(i_d, 9):
                            b_jj, b_9j, c, s = _givens(
                                float(b[j, j]), float(b[9, j])
                            )
                            b[j, j] = b_jj
                            b[9, j] = b_9j
                            if not (c == 1.0 and s == 0.0):
                                row_j = b[j, j + 1 : 10].copy()
                                row_9 = b[9, j + 1 : 10].copy()
                                b[j, j + 1 : 10] = c * row_j + s * row_9
                                b[9, j + 1 : 10] = -s * row_j + c * row_9
                    dmin_damped = min(
                        abs(float(b[d_idx, d_idx])) for d_idx in range(4, 9)
                    )
                    if dmin_damped * rc < _DTOL:
                        msg = "collinear linear subsystem in cshep2"
                        raise ValueError(msg)
                    break

                while True:
                    rsold = rs
                    neq += 1
                    if neq == lmax:
                        rc = math.sqrt(1.1 * rs)
                        break
                    if neq < lnp:
                        np_idx = int(npts[neq])
                        rs = (float(xv[np_idx - 1]) - xk) ** 2 + (
                            float(yv[np_idx - 1]) - yk
                        ) ** 2
                        if (rs - rsold) / rs < _RTOL:
                            continue
                        rc = math.sqrt(rs)
                        break
                    lnp += 1
                    np_idx, rs = _getnp2(
                        xk, yk, xv, yv, nr, lcell, lnext, xmin, ymin, dx, dy
                    )
                    npts[lnp - 1] = np_idx
                    if (rs - rsold) / rs < _RTOL:
                        continue
                    rc = math.sqrt(rs)
                    break

            sol = np.zeros(9, dtype=np.float64)
            for i in range(8, -1, -1):
                s_val = float(np.dot(b[i, i + 1 : 9], sol[i + 1 : 9]))
                sol[i] = (float(b[i, 9]) - s_val) / float(b[i, i])
            sol[0:4] *= sfc
            sol[4:7] *= sfs
            sol[7:9] *= sf
            a[k - 1, :] = sol

            lnext[k - 1] = -lnext[k - 1]
            for kp in range(lnp):
                i_unmark = int(npts[kp]) - 1
                lnext[i_unmark] = -lnext[i_unmark]

        return cls(
            x=xv,
            y=yv,
            f=fv,
            nr=nr,
            lcell=lcell,
            lnext=lnext,
            xmin=xmin,
            ymin=ymin,
            dx=dx,
            dy=dy,
            rmax=math.sqrt(rsmx),
            rw=rw,
            a=a,
        )

    def eval(self, px: float, py: float) -> float:
        """Evaluate $C(px, py)$ (`interp_cs2val_db`)."""
        return self.eval_hess(px, py)[0]

    def eval_grad(self, px: float, py: float) -> tuple[float, float, float]:
        """Evaluate $(C, C_x, C_y)$ (`interp_cs2grd_db`)."""
        c, cx, cy, _, _, _ = self.eval_hess(px, py)
        return c, cx, cy

    def eval_hess(
        self, px: float, py: float
    ) -> tuple[float, float, float, float, float, float]:
        """Evaluate $(C, C_x, C_y, C_{xx}, C_{xy}, C_{yy})$ (`cs2hes_db`)."""
        imin = max(int((px - self.xmin - self.rmax) / self.dx) + 1, 1)
        imax = min(int((px - self.xmin + self.rmax) / self.dx) + 1, self.nr)
        jmin = max(int((py - self.ymin - self.rmax) / self.dy) + 1, 1)
        jmax = min(int((py - self.ymin + self.rmax) / self.dy) + 1, self.nr)

        sw = swc = swx = swy = swcx = swcy = 0.0
        swxx = swxy = swyy = swcxx = swcxy = swcyy = 0.0

        for j in range(jmin, jmax + 1):
            for i in range(imin, imax + 1):
                k = int(self.lcell[i - 1, j - 1])
                if k == 0:
                    continue
                while True:
                    delx = px - float(self.x[k - 1])
                    dely = py - float(self.y[k - 1])
                    dxsq = delx * delx
                    dysq = dely * dely
                    d = math.sqrt(dxsq + dysq)
                    r = float(self.rw[k - 1])
                    if d < r:
                        ak = self.a[k - 1]
                        if d == 0.0:
                            return (
                                float(self.f[k - 1]),
                                float(ak[7]),
                                float(ak[8]),
                                2.0 * float(ak[4]),
                                float(ak[5]),
                                2.0 * float(ak[6]),
                            )
                        t1 = 1.0 / d - 1.0 / r
                        w = t1 * t1 * t1
                        t2 = -3.0 * t1 * t1 / (d * d * d)
                        wx, wy = delx * t2, dely * t2
                        t3 = 3.0 * t1 * (2.0 + 3.0 * d * t1) / (d**6)
                        wxx = t3 * dxsq + t2
                        wxy = t3 * delx * dely
                        wyy = t3 * dysq + t2

                        t1_c = (
                            float(ak[0]) * delx
                            + float(ak[1]) * dely
                            + float(ak[4])
                        )
                        t2_c = t1_c + t1_c + float(ak[0]) * delx
                        t3_c = (
                            float(ak[3]) * dely
                            + float(ak[2]) * delx
                            + float(ak[6])
                        )
                        t4_c = t3_c + t3_c + float(ak[3]) * dely
                        ck = (
                            (t1_c * delx + float(ak[5]) * dely + float(ak[7]))
                            * delx
                            + (t3_c * dely + float(ak[8])) * dely
                            + float(self.f[k - 1])
                        )
                        ckx = (
                            t2_c * delx
                            + (float(ak[2]) * dely + float(ak[5])) * dely
                            + float(ak[7])
                        )
                        cky = (
                            t4_c * dely
                            + (float(ak[1]) * delx + float(ak[5])) * delx
                            + float(ak[8])
                        )
                        ckxx = t2_c + 3.0 * float(ak[0]) * delx
                        ckxy = 2.0 * (
                            float(ak[1]) * delx + float(ak[2]) * dely
                        ) + float(ak[5])
                        ckyy = t4_c + 3.0 * float(ak[3]) * dely

                        sw += w
                        swc += w * ck
                        swx += wx
                        swy += wy
                        swcx += wx * ck + w * ckx
                        swcy += wy * ck + w * cky
                        swxx += wxx
                        swxy += wxy
                        swyy += wyy
                        swcxx += w * ckxx + 2.0 * wx * ckx + ck * wxx
                        swcxy += w * ckxy + wx * cky + wy * ckx + ck * wxy
                        swcyy += w * ckyy + 2.0 * wy * cky + ck * wyy
                    next_k = abs(int(self.lnext[k - 1]))
                    if next_k == k:
                        break
                    k = next_k

        if sw <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        c = swc / sw
        cx = (swcx - swx * c) / sw
        cy = (swcy - swy * c) / sw
        cxx = (swcxx - 2.0 * swx * cx - c * swxx) / sw
        cxy = (swcxy - swx * cy - swy * cx - c * swxy) / sw
        cyy = (swcyy - 2.0 * swy * cy - c * swyy) / sw
        return c, cx, cy, cxx, cxy, cyy


Qshep2Interpolant = Cshep2Interpolant


def interp_cshep2_db(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    nc: int = 17,
    nw: int = 32,
    nr: int = 6,
) -> Cshep2Interpolant:
    """Fit a scattered 2D cubic Shepard interpolant (`interp_cshep2_db`)."""
    return Cshep2Interpolant.fit(x, y, f, nc=nc, nw=nw, nr=nr)


def interp_cs2val_db(interp: Cshep2Interpolant, px: float, py: float) -> float:
    """Evaluate scattered 2D cubic Shepard interpolant (`interp_cs2val_db`)."""
    return interp.eval(px, py)


def interp_cs2grd_db(
    interp: Cshep2Interpolant, px: float, py: float
) -> tuple[float, float, float]:
    """Evaluate value and gradient (`interp_cs2grd_db`)."""
    return interp.eval_grad(px, py)


def cs2hes_db(
    interp: Cshep2Interpolant, px: float, py: float
) -> tuple[float, float, float, float, float, float]:
    """Evaluate value, gradient, and Hessian (`cs2hes_db`)."""
    return interp.eval_hess(px, py)


qshep2 = interp_cshep2_db
qs2val = interp_cs2val_db
qs2grd = interp_cs2grd_db
