# SPDX-License-Identifier: LGPL-3.0-or-later
"""``do_adjust_mass`` against MESA 26.4.1, zone by zone and bit for bit (#385).

Replays the setup of ``benchmarks/fortran_mesa/adjust_mass_oracle/``: a
4 Msun, Z = 0.02 ZAMS model with a composition step, then one mass removal
with region limits q <= 0.99 (kA) and q <= 0.95 (kB). ``q``, ``dq``, ``dm`` and
the H1 mass fraction must equal the MESA table before and after.
"""

from __future__ import annotations

import pathlib
import struct

import flexmesa
from flexmesa import constants as const
from flexmesa import star

_TABLE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks/fortran_mesa/adjust_mass_oracle/adjust_mass_4msun_26_4_1.tsv"
)


def _bits(x: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(x)))[0]:016X}"


def _mesa_rows() -> dict[tuple[str, int], list[str]]:
    rows: dict[tuple[str, int], list[str]] = {}
    for line in _TABLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("phase"):
            continue
        phase, k, *vals = line.split("\t")
        rows[(phase, int(k))] = vals
    return rows


def _snapshot(
    model: star.StarModel, phase: str
) -> dict[tuple[str, int], list[str]]:
    return {
        (phase, k + 1): [
            _bits(model.q[k]),
            _bits(model.dq[k]),
            _bits(model.dm[k]),
            _bits(model.xa[k, 0]),
        ]
        for k in range(model.nz)
    }


def test_adjust_mass_matches_mesa_26_4_1_zone_by_zone_bit_for_bit() -> None:
    mesa = _mesa_rows()
    with flexmesa.numerics_mode_context("strict"):
        z = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
        xa = z.xa.copy()
        for k in range(z.nz):
            if 0.95 < z.q[k] <= 0.98:
                xa[k, 0] -= 0.10
                xa[k, 2] += 0.10
        z._rust_model.set_primary_state(z.lnd, z.ln_t, z.ln_r, z.lum, z.dq, xa)
        ours = _snapshot(z, "BEFORE")
        mdot = z.apply_wind_and_adjust_mass(
            scheme="Dutch", eta=10.0, dt=1.0e6 * float(const.SECYER)
        )
        assert mdot.hex() == "-0x1.0640eab8de93dp+52"
        ours.update(_snapshot(z, "AFTER"))
    assert ours.keys() == mesa.keys()
    names = ("q", "dq", "dm", "xa_h1")
    for phase in ("BEFORE", "AFTER"):
        for j, name in enumerate(names):
            diff = [
                k
                for (p, k) in mesa
                if p == phase and ours[(p, k)][j] != mesa[(p, k)][j]
            ]
            assert not diff, (
                f"{phase} {name} differs from MESA 26.4.1 in {len(diff)} "
                f"zones, first k = {min(diff)}"
            )
