#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the REACLIB section of `crates/flexmesa-rates/data/rates_tables.bin`.

Reconstructs the REACLIB handle table of the `FXRT0001` bundle from an
unmodified MESA `26.4.1` tree (`$MESA_DIR`, Zenodo record `19722306`), following
`extract_rates_from_reaclib` (`rates/private/reaclib_input.f90:225`) with the
default `use_weaklib = .true.` and the full `chem_isos` nuclide set
(`rates/private/rates_initialize.f90:50-55`).

Inputs read from `$MESA_DIR`:

- `data/rates_data/jina_reaclib_results_20171020_default` (REACLIB rate sets)
- `data/rates_data/weakreactions.tables` (weaklib pairs)
- `data/chem_data/isotopes.data` (nuclide names, Z, N, spin, mass excess)

For every REACLIB set whose nuclides are all known, the exothermic direction
is kept (`reaclib_input.f90:307-313`), 1 -> 1 sets that weaklib already holds
are dropped (`reaclib_input.f90:316-340`), and the forward handle
(`get_reaction_handle`) and reverse handle (`get_reverse_reaction_handle`,
`rates/private/reaclib_support.f90:411-763`) are written with the same term
list; the reverse record carries `is_reverse = 1`. Records are sorted by
handle.

The other sections of the bundle (reaction metadata, `rate_tables`,
`weak_info`, weak and Suzuki tables) have no committed recipe; `--base` names
the existing bundle whose bytes outside the REACLIB section are carried over
unchanged, and the script refuses to run if that section cannot be located.
The loader recomputes the detailed-balance fields of non-weak terms at load
time (`crates/flexmesa-rates/src/loader.rs`), so those fields here only need
to be non-zero; they are still evaluated from `isotopes.data` for reference.

Usage:
    python scripts/data/build_rates_tables.py --mesa-dir $MESA_DIR \
        --base crates/flexmesa-rates/data/rates_tables.bin \
        --out crates/flexmesa-rates/data/rates_tables.bin
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import struct
from collections import defaultdict
from typing import Any

# Rows per `Nin`/`Nout` by REACLIB chapter (`rates/public/rates_def.f90`).
NIN = dict(enumerate([1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 1], start=1))
NOUT = dict(enumerate([1, 2, 3, 1, 2, 3, 4, 1, 2, 2, 4], start=1))

# `const_def` (MESA 26.4.1, CODATA 2018 / SI 2019).
PI = 3.1415926535897932384626433832795028841971693993751
AVO = 6.02214076e23
AMU = 1.0 / AVO
CLIGHT = 2.99792458e10
BOLTZM = 1.380649e-16
HBAR = 6.62607015e-27 / (2 * PI)
MEV_TO_ERGS = 1.0e6 * 1.602176634e-12

REACLIB_NAME_MAP = {
    "p": "h1",
    "n": "neut",
    "d": "h2",
    "t": "h3",
    "al-6": "al26-1",
    "al*6": "al26-2",
}
BUNDLE_MAGIC = b"FXRT\x01\x00\x00\x00"
NPART = 24


def sha256_file(path: pathlib.Path) -> str:
    """Return lowercase hex SHA-256 of `path`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Nuclides:
    """The `chem_isos` table from `isotopes.data`."""

    def __init__(self, path: pathlib.Path) -> None:
        lines = path.read_text().split("\n")[1:]  # line 1 is skipped upstream
        self.name: list[str] = []
        self.z: list[int] = []
        self.n: list[int] = []
        self.spin: list[float] = []
        self.mex: list[float] = []
        self.pfcn: list[list[float]] = []
        self.iso: list[int] = []
        i = 0
        while i + 3 < len(lines) and lines[i].strip():
            f = lines[i].split()
            self.name.append(f[0])
            self.z.append(int(f[2]))
            self.n.append(int(f[3]))
            self.spin.append(float(f[4]))
            self.mex.append(float(f[5]))
            pf: list[float] = []
            for k in range(1, 4):
                pf.extend(float(x) for x in lines[i + k].split())
            self.pfcn.append(pf)
            nm = f[0]
            self.iso.append(int(nm.split("-")[1]) if "-" in nm else 0)
            i += 4
        self.index = {nm: k for k, nm in enumerate(self.name)}
        mp = self.mex[self.index["h1"]]
        mn = self.mex[self.index["neut"]]
        # `del_Mp` / `del_Mn` are taken from the last (Z,N) match, as upstream.
        for k in range(len(self.name)):
            if self.z[k] == 1 and self.n[k] == 0:
                mp = self.mex[k]
            if self.n[k] == 1 and self.z[k] == 0:
                mn = self.mex[k]
        self.be = [
            self.z[k] * mp + self.n[k] * mn - self.mex[k]
            for k in range(len(self.name))
        ]
        self.w = [
            (self.z[k] + self.n[k])
            + self.mex[k] * (MEV_TO_ERGS / AMU / (CLIGHT * CLIGHT))
            for k in range(len(self.name))
        ]


def parse_reaclib(path: pathlib.Path) -> list[dict[str, Any]]:
    """Parse REACLIB sets (`do_read_reaclib`, `reaclib_input.f90:72`)."""
    lines = path.read_text().split("\n")
    out: list[dict[str, Any]] = []
    i = 0
    while i + 3 < len(lines) and lines[i].strip():
        chapter = int(lines[i])
        l1 = lines[i + 1]
        species = [l1[5 + 5 * k : 10 + 5 * k].strip() for k in range(6)]
        species = [REACLIB_NAME_MAP.get(s, s) for s in species]
        co: list[float] = []
        for s in (lines[i + 2], lines[i + 3]):
            for a, b in ((0, 13), (13, 26), (26, 39), (39, 52)):
                if s[a:b].strip():
                    co.append(float(s[a:b]))
        out.append(
            {
                "chapter": chapter,
                "species": species,
                "label": l1[43:47].strip(),
                "flag": l1[47:48],
                "coeffs": co,
            }
        )
        i += 4
    return out


def parse_weak_pairs(path: pathlib.Path) -> set[tuple[str, str]]:
    """Return the (lhs, rhs) names of weaklib tables (`load_weak.f90`).

    Each table in `weakreactions.tables` starts with a line
    `<lhs> <rhs>   by ...; <source>`; the same lines also form the index
    near the top of the file.
    """
    pairs: set[tuple[str, str]] = set()
    for line in path.read_text().split("\n"):
        f = line.split()
        if len(f) >= 4 and f[2] == "by" and ";" in line:
            a, b = (REACLIB_NAME_MAP.get(x, x) for x in f[:2])
            pairs.add((a, b))
    # `S13_r_be7_wk_li7.h5` adds be7 -> li7 (`load_weak.f90`).
    pairs.add(("be7", "li7"))
    return pairs


def handle_names(
    nuc: Nuclides,
    num_in: int,
    num_out: int,
    pspecies: list[int],
    reverse: bool,
    flag: str,
) -> str:
    """Port of `get1_reaction_handle` (`reaclib_support.f90:411`)."""
    z, n, name = nuc.z, nuc.n, nuc.name
    num = num_in + num_out
    ps = list(pspecies[:num])

    def key(c: int) -> tuple[int, int, int]:
        return (z[c], n[c], nuc.iso[c])

    def sort(a: int, b: int) -> None:
        sub = ps[a:b]
        # Upstream's exchange sort is stable for equal keys.
        sub.sort(key=key)
        ps[a:b] = sub

    sort(0, num_in)
    sort(num_in, num)
    ec = flag == "e"
    wk = flag == "w"
    ih1, ih2, ih3 = nuc.index["h1"], nuc.index["h2"], nuc.index["h3"]
    ihe3, ibe7, ili7 = nuc.index["he3"], nuc.index["be7"], nuc.index["li7"]
    if ec:
        if reverse:
            return ""
        if num_in == 2 and num_out == 1:
            if ps[0] == ih1 and ps[1] == ih1 and ps[2] == ih2:
                return "r_h1_h1_ec_h2"
        elif num_in == 1 and num_out == 1:
            if ps[0] == ihe3 and ps[1] == ih3:
                return "r_he3_ec_h3"
            if ps[0] == ibe7 and ps[1] == ili7:
                return "r_be7_wk_li7"
    elif wk:
        if reverse:
            return ""
        if num_in == 2 and num_out == 1:
            if ps[0] == ih1 and ps[1] == ih1 and ps[2] == ih2:
                return "r_h1_h1_wk_h2"

    in1 = in2 = out1 = out2 = -1
    handle = ""

    def is_(c: int, zz: int, nn: int) -> bool:
        return z[c] == zz and n[c] == nn

    def switch(i1: int, i2: int) -> tuple[int, int]:
        if is_(i2, 1, 0):
            return i2, i1
        if is_(i2, 2, 2):
            if is_(i1, 1, 0):
                return i1, i2
            return i2, i1
        return i1, i2

    def n_to_m(nn: int, mm: int) -> None:
        nonlocal in1, in2, out1, out2, handle
        in1 = in2 = out1 = out2 = -1
        if not reverse:
            in1 = ps[0]
            if nn == 2:
                in2 = ps[1]
                in1, in2 = switch(in1, in2)
                handle = "r_" + name[in2]
            else:
                handle = "r_" + name[in1]
            out1 = ps[nn]
            if mm == 2:
                out2 = ps[nn + 1]
                out1, out2 = switch(out1, out2)
        else:
            in1 = ps[nn]
            if mm == 2:
                in2 = ps[nn + 1]
                in1, in2 = switch(in1, in2)
                handle = "r_" + name[in2]
            else:
                handle = "r_" + name[in1]
            out1 = ps[0]
            if nn == 2:
                out2 = ps[1]
                out1, out2 = switch(out1, out2)

    def one_one() -> bool:
        nonlocal handle
        if in1 < 0 or out1 < 0:
            return True
        if z[out1] == z[in1] - 1 and n[out1] == n[in1] + 1:
            handle += "_wk"
        elif z[out1] == z[in1] + 1 and n[out1] == n[in1] - 1:
            handle += "_wk-minus"
        else:
            return True
        return False

    def one_two() -> bool:
        nonlocal handle
        if in1 < 0 or out1 < 0 or out2 < 0 or out1 == out2:
            return True
        zi, ni = z[in1], n[in1]
        if is_(out1, 0, 1) and z[out2] == zi and n[out2] == ni - 1:
            handle += "_gn"
        elif is_(out1, 1, 0) and z[out2] == zi - 1 and n[out2] == ni:
            handle += "_gp"
        elif is_(out1, 2, 2) and z[out2] == zi - 2 and n[out2] == ni - 2:
            handle += "_ga"
        elif is_(out1, 1, 0) and z[out2] == zi - 2 and n[out2] == ni + 1:
            handle += "_wk_h1"
        elif is_(out1, 2, 2) and z[out2] == zi - 3 and n[out2] == ni - 1:
            handle += "_wk_he4"
        else:
            return True
        return False

    def two_one() -> bool:
        nonlocal handle
        if in1 < 0 or in2 < 0 or out1 < 0 or in1 == in2:
            return True
        if is_(in1, 0, 1) and z[out1] == z[in2] and n[out1] == n[in2] + 1:
            handle += "_ng"
        elif is_(in1, 1, 0) and z[out1] == z[in2] + 1 and n[out1] == n[in2]:
            handle += "_pg"
        elif is_(in1, 2, 2) and z[out1] == z[in2] + 2 and n[out1] == n[in2] + 2:
            handle += "_ag"
        else:
            return True
        return False

    def two_two() -> bool:
        nonlocal handle
        if in1 < 0 or in2 < 0 or out1 < 0 or out2 < 0:
            return True
        if (
            is_(in1, 1, 0)
            and is_(out1, 2, 2)
            and is_(out2, 2, 2)
            and is_(in2, 3, 4)
        ):
            handle += "_pa"
            return False
        if in1 == in2 or out1 == out2:
            return True
        zi, ni = z[in2], n[in2]
        if (
            is_(in1, 2, 2)
            and is_(out1, 1, 0)
            and z[out2] == zi + 1
            and n[out2] == ni + 2
        ):
            handle += "_ap"
        elif (
            is_(in1, 1, 0)
            and is_(out1, 2, 2)
            and z[out2] == zi - 1
            and n[out2] == ni - 2
        ):
            handle += "_pa"
        elif (
            is_(in1, 2, 2)
            and is_(out1, 0, 1)
            and z[out2] == zi + 2
            and n[out2] == ni + 1
        ):
            handle += "_an"
        elif (
            is_(in1, 0, 1)
            and is_(out1, 2, 2)
            and z[out2] == zi - 2
            and n[out2] == ni - 1
        ):
            handle += "_na"
        elif (
            is_(in1, 1, 0)
            and is_(out1, 0, 1)
            and z[out2] == zi + 1
            and n[out2] == ni - 1
        ):
            handle += "_pn"
        elif (
            is_(in1, 0, 1)
            and is_(out1, 1, 0)
            and z[out2] == zi - 1
            and n[out2] == ni + 1
        ):
            handle += "_np"
        else:
            return True
        return False

    long_form = True
    if num_in == 1 and num_out == 1:
        n_to_m(1, 1)
        long_form = one_one()
    elif num_in == 1 and num_out == 2:
        n_to_m(1, 2)
        long_form = two_one() if reverse else one_two()
    elif num_in == 2 and num_out == 1:
        n_to_m(2, 1)
        long_form = one_two() if reverse else two_one()
    elif num_in == 2 and num_out == 2:
        n_to_m(2, 2)
        long_form = two_two()

    if long_form:
        op = "wk_" if wk else ("ec_" if ec else "to_")
        handle = "r_"
        if reverse:
            for c in ps[num_in:num]:
                handle += name[c] + "_"
            handle += op
            handle += "_".join(name[c] for c in ps[:num_in])
        else:
            for c in ps[:num_in]:
                handle += name[c] + "_"
            handle += op
            handle += "_".join(name[c] for c in ps[num_in:num])
    elif out2 >= 0:
        handle += "_" + name[out2]
    else:
        handle += "_" + name[out1]
    return handle


def detailed_balance(
    nuc: Nuclides, ni: int, no: int, ps: list[int]
) -> tuple[float, float, list[float]]:
    """`compute_rev_ratio` (`reaclib_support.f90:172`) for one term."""
    nt = ni + no
    fac = math.pow(1.0e9 * BOLTZM / (2.0 * PI * HBAR * HBAR * AVO), 1.5) / AVO
    g = [2.0 * nuc.spin[c] + 1.0 for c in ps[:nt]]
    mass = [nuc.w[c] for c in ps[:nt]]
    num = 1.0
    for m in mass[:ni]:
        num *= m
    den = 1.0
    for m in mass[ni:nt]:
        den *= m
    c1 = math.pow(num / den, 1.5)
    gp = 1.0
    for x in g[:ni]:
        gp *= x
    gq = 1.0
    for x in g[ni:nt]:
        gq *= x
    c1 *= gp / gq
    s1 = sum(nuc.be[c] for c in ps[:ni])
    s2 = sum(nuc.be[c] for c in ps[ni:nt])
    c2 = (s1 - s2) * MEV_TO_ERGS / BOLTZM / 1.0e9
    inv_exp = ni - no
    if inv_exp != 0:
        c1 *= math.pow(fac, inv_exp)
    else:
        inv_exp = 0
    c1 = math.log(c1)
    part = []
    for k in range(NPART):
        a = 1.0
        for c in ps[:ni]:
            a *= nuc.pfcn[c][k]
        b = 1.0
        for c in ps[ni:nt]:
            b *= nuc.pfcn[c][k]
        part.append(a / b)
    return c1, c2, part


def build_handles(
    nuc: Nuclides,
    sets: list[dict[str, Any]],
    weak_pairs: set[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    """Return `{handle: {"rev": bool, "terms": [...]}}` for all kept sets."""
    handles: dict[str, dict[str, Any]] = {}
    limits: dict[str, list[int]] = {}
    weak_sourced: dict[str, dict[str, Any]] = {}
    fwd_terms: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for e in sets:
        ch = e["chapter"]
        ni, no = NIN[ch], NOUT[ch]
        nt = ni + no
        ps = []
        ok = True
        for s in e["species"][:nt]:
            if s not in nuc.index:
                ok = False
                break
            ps.append(nuc.index[s])
        if not ok:
            continue
        is_weak = ni == 1 and no == 1
        if not is_weak:
            if (
                sum(nuc.be[c] for c in ps[:ni])
                - sum(nuc.be[c] for c in ps[ni:nt])
                > 0
            ):
                continue
        if is_weak:
            a, b = e["species"][0], e["species"][1]
            for lhs, rhs in ((a, b), (b, a)):
                if (lhs, rhs) in weak_pairs:
                    # Upstream stores the coefficients of the last matching
                    # set on the weaklib-sourced reaction
                    # (`reaclib_input.f90:316-340`), named in weaklib order.
                    wps = [nuc.index[lhs], nuc.index[rhs]]
                    wh = handle_names(nuc, 1, 1, wps, False, "")
                    ie = 0
                    lim = limits.setdefault(wh, [0, 0])
                    lim[0] = max(lim[0], nuc.z[wps[0]], nuc.z[wps[1]])
                    lim[1] = max(lim[1], nuc.z[wps[0]] + nuc.n[wps[0]])
                    coeffs = e["coeffs"]
                    if "h1" in (lhs, rhs) or "neut" in (lhs, rhs):
                        # The free-nucleon weaklib reactions end up with
                        # the `-99` placeholder in MESA (existing oracle:
                        # `rates_161_parity_test`, exp(-99) rate). Upstream
                        # does store the n->p REACLIB coefficients on
                        # weaklib entry 1, but the lookup lands on entry 2
                        # because `prot` duplicates `h1` in `chem_isos` and
                        # the lo..hi scan never sets lo = 1
                        # (`do_reaclib_indices_for_reaction`). This rule
                        # reproduces that outcome; it is not the mechanism.
                        coeffs = [-99.0]
                    weak_sourced[wh] = {
                        "rev": False,
                        "terms": [
                            {
                                "chapter": 1,
                                "flag": "w",
                                "coeffs": coeffs,
                                "inv": (
                                    -1.7976931348623157e308,
                                    0.0,
                                    [1.0] * NPART,
                                ),
                                "inv_exp": ie,
                            }
                        ],
                    }
                    break
            if (a, b) in weak_pairs or (b, a) in weak_pairs:
                continue
        flag = e["flag"]
        if e["label"].startswith("ec"):
            flag = "e"
        h = handle_names(nuc, ni, no, ps, False, flag)
        if flag in ("e", "w"):
            rh = ""
        else:
            rh = handle_names(nuc, ni, no, ps, True, "-")
        term = {
            "chapter": ch,
            "flag": flag if flag.strip() else "-",
            "coeffs": e["coeffs"],
        }
        if flag in ("e", "w"):
            term["inv"] = (-1.7976931348623157e308, 0.0, [1.0] * NPART)
            term["inv_exp"] = 0
        else:
            c1, c2, part = detailed_balance(nuc, ni, no, ps)
            term["inv"] = (c1, c2, part)
            ie = ni - no
            term["inv_exp"] = ie if ie != 0 else 0
        fwd_terms[h].append(term)
        order.append((h, rh))
        zmax_a = max(nuc.z[c] for c in ps)
        amax_a = max(nuc.z[c] + nuc.n[c] for c in ps)
        for name in (h, rh):
            if name:
                lim = limits.setdefault(name, [0, 0])
                lim[0] = max(lim[0], zmax_a)
                lim[1] = max(lim[1], amax_a)
    for h, rh in order:
        if h not in handles:
            handles[h] = {"rev": False, "terms": fwd_terms[h]}
        if rh and rh not in handles:
            handles[rh] = {"rev": True, "terms": fwd_terms[h]}
    for h, rec in weak_sourced.items():
        handles.setdefault(h, rec)
    for h, (zm, am) in limits.items():
        if h in handles:
            handles[h]["max_z"] = zm
            handles[h]["max_a"] = am
    return handles


def serialize_record(h: str, rec: dict[str, Any]) -> bytes:
    """Serialize one handle record (`loader.rs` `parse_builtin`)."""
    hb = h.encode()
    terms = rec["terms"]
    buf = bytearray(
        struct.pack("<HBB", len(hb), int(rec["rev"]), len(terms)) + hb
    )
    for t in terms:
        co = list(t["coeffs"]) + [0.0] * (7 - len(t["coeffs"]))
        c1, c2, part = t["inv"]
        buf += struct.pack(
            "<BBb7d2d24d",
            t["chapter"],
            ord(t["flag"]),
            t["inv_exp"],
            *co,
            c1,
            c2,
            *part,
        )
    return bytes(buf)


def base_records(blob: bytes, start: int, end: int) -> dict[str, bytes]:
    """Return the raw per-handle records of the base REACLIB section."""
    pos = start + 4
    out: dict[str, bytes] = {}
    while pos < end:
        hl, _r, nt = struct.unpack_from("<HBB", blob, pos)
        size = 4 + hl + nt * (3 + 8 * (7 + 2 + NPART))
        out[blob[pos + 4 : pos + 4 + hl].decode()] = blob[pos : pos + size]
        pos += size
    return out


def locate_section(blob: bytes) -> tuple[int, int, int]:
    """Return `(start, end, n_handles)` of the REACLIB section of a bundle."""
    if not blob.startswith(BUNDLE_MAGIC):
        raise ValueError("not an FXRT0001 bundle")
    pos = 8
    (nr,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    for _ in range(nr):
        nl, _op, _rev, tl = struct.unpack_from("<HBBH", blob, pos)
        pos += 6 + nl + tl + 1 + 3 + 6 + 1 + 3 + 6 + 3 + 6 + 4 + 24
    (nrt,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    for _ in range(nrt):
        (nt8,) = struct.unpack_from("<2xH", blob, pos)
        pos += 4 + 8 * nt8 + 32 * nt8
    start = pos
    (nh,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    for _ in range(nh):
        hl, _r, nt = struct.unpack_from("<HBB", blob, pos)
        pos += 4 + hl + nt * (3 + 8 * (7 + 2 + NPART))
    return start, pos, nh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    ap.add_argument("--base", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--list-handles",
        type=pathlib.Path,
        help="Also write the handles with Z <= --max-z and A <= --max-a.",
    )
    ap.add_argument("--max-z", type=int, default=10)
    ap.add_argument("--max-a", type=int, default=26)
    ap.add_argument(
        "--reaclib",
        default="jina_reaclib_results_20171020_default",
        help="REACLIB file under data/rates_data.",
    )
    args = ap.parse_args()
    rd = args.mesa_dir / "data" / "rates_data"
    nuc = Nuclides(args.mesa_dir / "data" / "chem_data" / "isotopes.data")
    sets = parse_reaclib(rd / args.reaclib)
    weak = parse_weak_pairs(rd / "weakreactions.tables")
    handles = build_handles(nuc, sets, weak)
    if args.list_handles is not None:
        sel = sorted(
            h
            for h, rec in handles.items()
            if rec.get("max_z", 999) <= args.max_z
            and rec.get("max_a", 999) <= args.max_a
        )
        args.list_handles.parent.mkdir(parents=True, exist_ok=True)
        args.list_handles.write_text("\n".join(sel) + "\n")
        print(f"listed {len(sel)} handles to {args.list_handles}")
    base = args.base.read_bytes()
    start, end, n_old = locate_section(base)
    records = {h: serialize_record(h, rec) for h, rec in handles.items()}
    # `r_h1_wk_neut` is a weaklib-sourced record with placeholder
    # coefficients (`-99`, two terms) that the base bundle carries; it is
    # kept byte for byte.
    carried = {
        h: raw
        for h, raw in base_records(base, start, end).items()
        if h not in records
    }
    records.update(carried)
    section = struct.pack("<I", len(records)) + b"".join(
        records[h] for h in sorted(records)
    )
    new = base[:start] + section + base[end:]
    args.out.write_bytes(new)
    print(f"reaclib sets read: {len(sets)}")
    print(f"handles: {n_old} -> {len(records)} (carried: {sorted(carried)})")
    print(f"bytes: {len(base)} -> {len(new)}")
    for p in (
        rd / args.reaclib,
        rd / "weakreactions.tables",
        args.mesa_dir / "data" / "chem_data" / "isotopes.data",
    ):
        print(f"source {p.relative_to(args.mesa_dir)} sha256={sha256_file(p)}")
    print(f"out sha256={sha256_file(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
