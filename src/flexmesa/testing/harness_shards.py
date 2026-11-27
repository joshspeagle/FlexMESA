# SPDX-License-Identifier: GPL-3.0-or-later
"""Split a harness campaign into shards and merge the shard reports.

A full strict campaign runs every manifest case once. Strict mode is
bit-identical across machines and thread counts, so the cases can run on
several machines at once and the per-case results can be combined into one
case table and one baseline candidate afterwards. Only the outcomes cut off
by a time or memory budget depend on the machine.

Each shard writes the usual ``--json-out`` report with a ``run`` block that
records the shard and the run settings; :func:`merge_shard_payloads` checks
that the shards form one complete campaign with uniform settings.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from flexmesa.testing import case_manifest

__all__ = [
    "RUN_SETTING_KEYS",
    "MergedShards",
    "merge_shard_payloads",
    "parse_shard",
    "shard_entries",
]

#: Keys of a report's ``run`` block that must agree across shards.
RUN_SETTING_KEYS: tuple[str, ...] = (
    "numerics_mode",
    "max_steps",
    "allow_unsupported_controls",
    "timeout_sec",
    "short_budget_sec",
    "max_rss_gb",
    "budget_from_baseline",
    "case_budgets",
    "mesa_step_factor",
    "early_stop_on_divergence",
    "stop_on_excess_steps",
    "enforce_step_counts",
)


def parse_shard(raw: str) -> tuple[int, int]:
    """Parse a ``"I/N"`` shard specification.

    Args:
        raw: Shard text, with ``1 <= I <= N``.

    Returns:
        ``(index, count)`` with a 1-based ``index``.

    Raises:
        ValueError: If ``raw`` is not of the form ``I/N`` with
            ``1 <= I <= N``.
    """
    parts = raw.split("/")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"shard must be 'I/N', got {raw!r}")
    index, count = int(parts[0]), int(parts[1])
    if not 1 <= index <= count:
        raise ValueError(f"shard index must satisfy 1 <= I <= N, got {raw!r}")
    return index, count


def shard_entries(
    entries: list[case_manifest.CaseManifestEntry],
    index: int,
    count: int,
    costly: frozenset[tuple[str, str]] = frozenset(),
    weights: Mapping[tuple[str, str], float] | None = None,
) -> list[case_manifest.CaseManifestEntry]:
    """Select the entries one shard runs.

    With ``weights`` the entries are packed by weight: heaviest first, each
    onto the shard with the smallest total so far (lowest index on a tie),
    which keeps the heaviest shard close to the mean. An entry without a
    weight counts as the largest weight given, so an unlisted case is never
    assumed cheap.

    Without ``weights`` the entries are ordered with the ``costly`` cases
    (those expected to run to their time or memory budget) first, each group
    in its input order, and dealt round-robin to the shards, so every shard
    gets a near-equal share of the costly cases. Either way the assignment
    depends only on the inputs.

    Args:
        entries: Selected entries in manifest order.
        index: 1-based shard index.
        count: Number of shards.
        costly: ``(suite, case)`` keys dealt out first (ignored when
            ``weights`` is given).
        weights: Optional cost per ``(suite, case)``, such as the wall-time
            budget in seconds (``s``).

    Returns:
        The entries of shard ``index``, in manifest order.
    """
    if weights:
        heaviest = max(weights.values())
        cost = {e.key: weights.get(e.key, heaviest) for e in entries}
        load = [0.0] * count
        mine = set()
        for e in sorted(entries, key=lambda e: (-cost[e.key], e.key)):
            slot = min(range(count), key=lambda i: (load[i], i))
            load[slot] += cost[e.key]
            if slot == index - 1:
                mine.add(e.key)
        return [e for e in entries if e.key in mine]
    ordered = [e for e in entries if e.key in costly] + [
        e for e in entries if e.key not in costly
    ]
    mine = {e.key for e in ordered[index - 1 :: count]}
    return [e for e in entries if e.key in mine]


@dataclasses.dataclass(frozen=True)
class MergedShards:
    """Case rows and run settings combined from shard reports.

    Attributes:
        rows: Executed case rows keyed by ``(suite, case)``.
        settings: The run settings shared by every shard.
        short_budget_cases: ``(suite, case)`` keys run with the short budget.
        budgeted_cases: ``(suite, case)`` keys that ran with a wall-time
            budget derived from the baseline (``--budget-from-baseline``).
        problems: Reasons the shards do not form one complete campaign
            (missing or repeated shards, repeated cases); empty when they
            do.
    """

    rows: dict[tuple[str, str], dict[str, Any]]
    settings: dict[str, Any]
    short_budget_cases: frozenset[tuple[str, str]]
    budgeted_cases: frozenset[tuple[str, str]]
    problems: tuple[str, ...]


def _label_key(label: str) -> tuple[str, str]:
    suite, _, case = label.partition("/")
    return suite, case


def merge_shard_payloads(payloads: list[dict[str, Any]]) -> MergedShards:
    """Combine the ``--json-out`` reports of the shards of one campaign.

    Args:
        payloads: Parsed shard reports, each with a ``run`` block.

    Returns:
        The combined rows and settings, with any completeness problems.

    Raises:
        ValueError: If there are no payloads, a payload has no ``run`` block
            or shard, or the shards disagree on a run setting or on the
            shard count.
    """
    if not payloads:
        raise ValueError("no shard reports to merge")
    runs: list[dict[str, Any]] = []
    for pos, payload in enumerate(payloads):
        run = payload.get("run")
        if not isinstance(run, dict) or not run.get("shard"):
            raise ValueError(f"report {pos} has no 'run.shard' block")
        runs.append(run)
    settings = {k: runs[0].get(k) for k in RUN_SETTING_KEYS}
    for run in runs[1:]:
        for key in RUN_SETTING_KEYS:
            if run.get(key) != settings[key]:
                raise ValueError(
                    f"shards disagree on {key}: {settings[key]!r} vs "
                    f"{run.get(key)!r}"
                )
    counts = {int(run["shard"][1]) for run in runs}
    if len(counts) != 1:
        raise ValueError(f"shards disagree on the shard count: {counts}")
    (count,) = counts
    problems: list[str] = []
    seen = [int(run["shard"][0]) for run in runs]
    missing = sorted(set(range(1, count + 1)) - set(seen))
    if missing:
        problems.append(f"missing shards {missing} of {count}")
    repeated = sorted({i for i in seen if seen.count(i) > 1})
    if repeated:
        problems.append(f"repeated shards {repeated}")

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    short: set[tuple[str, str]] = set()
    budgeted: set[tuple[str, str]] = set()
    for payload, run in zip(payloads, runs, strict=True):
        short.update(_label_key(c) for c in run.get("short_budget_cases", ()))
        budgeted.update(_label_key(c) for c in run.get("budgeted_cases", ()))
        for row in payload.get("cases", ()):
            if not row.get("executed"):
                continue
            row_key = (str(row["suite"]), str(row["case"]))
            if row_key in rows:
                problems.append(f"case {row_key[0]}/{row_key[1]} in two shards")
            rows[row_key] = row
        for label in run.get("selected_cases", ()):
            if _label_key(label) not in rows:
                problems.append(f"case {label} selected but not executed")
    return MergedShards(
        rows=rows,
        settings=settings,
        short_budget_cases=frozenset(short),
        budgeted_cases=frozenset(budgeted),
        problems=tuple(problems),
    )
