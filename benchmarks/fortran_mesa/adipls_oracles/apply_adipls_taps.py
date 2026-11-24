#!/usr/bin/env python3
"""Insert read-only intermediate taps into an overlay copy of MESA's ADIPLS.

Usage: apply_adipls_taps.py <overlay adipls dir>

Edits fixed-form sources under <dir>/adipack.c/adipls of an overlay copy (the
pristine tree is never touched). Each tap is one `call a0trec(tag, n, values)`
statement (adipls_tap.c) that formats its arguments as hex; it writes no
variable ADIPLS uses. Every anchor must match exactly once (#171, package A0).

Tags (record name in the tap file):
  0 ENTER   adipls.c.d.f, label 10000: start of a per-model call
  1 DET     sigint.n.d.f, exit: sig, det, ddsol, iord, icry, nfit for every
            integration (scan points first, then iterates of sigitr)
  2 MATCH   mchsol.n.d.f, normal exit: det, y(1:ii, nfit) after matching
  3 ITC     sigitc.c.d.f, exit: sig, ddsig, ddsol, xfit1, nfit, iord, icry,
            isolcv, isigcv, istatus (converged mode)
  4 KINER   kiner.n.d.f, exit: el, ynorm, ekin
  5 VARFRQ  varfrq.n.d.f, exit: sig, sigv (variational frequency)
  6 GRAVPO  gravpo.n.d.f, normal exit: sig, dsig
"""

from __future__ import annotations

import pathlib
import sys

EDITS: list[tuple[str, str, str]] = [
    # (file, anchor, replacement); the replacement contains the anchor
    (
        "adipls.c.d.f",
        "10000 continue\n",
        "10000 continue\n      call a0trec(0, 0, 0.d0)\n",
    ),
    (
        "sigint.n.d.f",
        "      end if\n      return\n      end\n",
        "      end if\n"
        "      call a0trec(1, 6, (/sig, det, ddsol, dble(iord), dble(icry),\n"
        "     *  dble(nfit)/))\n"
        "      return\n      end\n",
    ),
    (
        "mchsol.n.d.f",
        "      icry = 0\nc\n      return\n",
        "      icry = 0\n"
        "      call a0trec(2, 1+ii, (/det, (y(ia0t,nfit), ia0t=1,ii)/))\n"
        "c\n      return\n",
    ),
    (
        "sigitc.c.d.f",
        "        call sclasl(x,y,iy,nn,ii,nev1,nw1,el,iplneq)\n      end if\nc\n      return\n",
        "        call sclasl(x,y,iy,nn,ii,nev1,nw1,el,iplneq)\n      end if\n"
        "      call a0trec(3, 10, (/sig, ddsig, ddsol, xfit1, dble(nfit),\n"
        "     *  dble(iord), dble(icry), dble(isolcv), dble(isigcv),\n"
        "     *  dble(istatus)/))\n"
        "c\n      return\n",
    ),
    (
        "kiner.n.d.f",
        "      ekin=wrk(2,nn)/(4.d0*pi*ynorm)\n      return\n",
        "      ekin=wrk(2,nn)/(4.d0*pi*ynorm)\n"
        "      call a0trec(4, 3, (/el, ynorm, ekin/))\n"
        "      return\n",
    ),
    (
        "varfrq.n.d.f",
        "c\n      return\n   90 format(//' variational integral, formulation',i3)\n",
        "      call a0trec(5, 2, (/sig, sigv/))\n"
        "c\n      return\n   90 format(//' variational integral, formulation',i3)\n",
    ),
    (
        "gravpo.n.d.f",
        "      nwrt=0\n      return\n",
        "      nwrt=0\n      call a0trec(6, 2, (/sig, dsig/))\n      return\n",
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = pathlib.Path(sys.argv[1])
    if not (root / ".flexmesa_adipls_overlay").exists():
        print(f"refusing: {root} is not marked as an overlay copy", file=sys.stderr)
        return 2
    src = root / "adipack.c" / "adipls"
    texts: dict[str, str] = {}
    for fname, anchor, repl in EDITS:
        s = texts.get(fname) or (src / fname).read_text()
        n = s.count(anchor)
        if n != 1:
            print(f"anchor matched {n} times in {fname}: {anchor!r}", file=sys.stderr)
            return 1
        texts[fname] = s.replace(anchor, repl)
    for fname, s in texts.items():
        for ln in s.splitlines():
            if "a0trec" in ln or ln.startswith("     *  d"):
                assert len(ln) <= 72, f"fixed-form line too long in {fname}: {ln!r}"
        (src / fname).write_text(s)
    print(f"tapped {len(EDITS)} sites in {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
