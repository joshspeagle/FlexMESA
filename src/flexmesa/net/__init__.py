# SPDX-License-Identifier: LGPL-3.0-or-later
"""Nuclear reaction networks, analytical Jacobians, and burners (`net`)."""

from __future__ import annotations

from flexmesa.net import _core, _jax

Burn1ZoneOutput = _core.Burn1ZoneOutput
BurnConstPOutput = _core.BurnConstPOutput
NetEvalOutput = _core.NetEvalOutput
NetInfo = _core.NetInfo
burn_1_zone = _core.burn_1_zone
burn_1_zone_const_p = _core.burn_1_zone_const_p
eval_net = _core.eval_net
eval_net_jax = _jax.eval_net_jax
get_net_info = _core.get_net_info
list_networks = _core.list_networks

__all__: list[str] = [
    "Burn1ZoneOutput",
    "BurnConstPOutput",
    "NetEvalOutput",
    "NetInfo",
    "burn_1_zone",
    "burn_1_zone_const_p",
    "eval_net",
    "eval_net_jax",
    "get_net_info",
    "list_networks",
]
