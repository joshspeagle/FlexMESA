# SPDX-License-Identifier: GPL-3.0-or-later
"""GitHub tracker and pull-request snapshot collector (`#174`).

Reads `#140` for the work phases, the deferred register tables of every
child tracker of the Deferred super-tracker `#727`, and the open pull
requests.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import urllib.request
from collections.abc import Sequence
from typing import Any

#: GitHub caps a parent issue at 100 sub-issues, so the trackers sit under
#: super-trackers: `#725` Work (holds `#140`), `#726` Parity (`#546`, `#722`)
#: and `#727` Deferred (`#169`, `#728`). The collector reads the deferred
#: register from every child tracker of `#727`, so a new continuation tracker
#: needs no code change. `#725` and `#726` are named for reference: `#140`'s
#: phases are parsed as before, and this collector counts no `[Parity]` issue.
WORK_SUPER_TRACKER = 725
PARITY_SUPER_TRACKER = 726
DEFERRED_SUPER_TRACKER = 727

_PHASE_ORDER: tuple[str, ...] = (
    "Phase 0",
    "Phase A",
    "Phase B",
    "Phase C",
    "Phase D",
    "Phase E",
    "Deferred scope",
)


def _gh_api_json(endpoint: str) -> Any:
    """Fetch JSON from GitHub REST API via ``GITHUB_TOKEN`` or ``gh api``."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        url = f"https://api.github.com/{endpoint.lstrip('/')}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "flexmesa-dashboard",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    proc = subprocess.run(
        ["gh", "api", endpoint],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def _parse_140_phases(
    body_140: str, issues_by_num: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group `#140` sub-issues by phase and append `deferred-scope` issues."""
    phase_issues: dict[str, list[int]] = {p: [] for p in _PHASE_ORDER}
    current_phase: str | None = None
    in_work_packages = False
    for line in body_140.splitlines():
        if line.startswith("## Work packages"):
            in_work_packages = True
            continue
        if in_work_packages and line.startswith("## "):
            break
        if not in_work_packages:
            continue
        if line.startswith("### "):
            header = line[4:].strip()
            current_phase = next(
                (p for p in _PHASE_ORDER[:6] if header.startswith(p)), None
            )
            continue
        if current_phase is not None and line.lstrip().startswith("- ["):
            m = re.search(r"#(\d+)\b", line)
            if m is not None:
                inum = int(m.group(1))
                if inum not in phase_issues[current_phase]:
                    phase_issues[current_phase].append(inum)

    seen_140: set[int] = set()
    for nums in phase_issues.values():
        seen_140.update(nums)
    for num, item in sorted(issues_by_num.items()):
        labels = {str(lb["name"]) for lb in item.get("labels", [])}
        if "deferred-scope" in labels and num not in seen_140:
            phase_issues["Deferred scope"].append(num)
            seen_140.add(num)

    phases_payload: list[dict[str, Any]] = []
    for pname in _PHASE_ORDER:
        items_out = [
            {
                "number": inum,
                "title": str(issues_by_num[inum]["title"]),
                "state": str(issues_by_num[inum]["state"]),
            }
            for inum in sorted(phase_issues[pname])
            if inum in issues_by_num
        ]
        phases_payload.append(
            {
                "phase": pname,
                "open": sum(1 for x in items_out if x["state"] == "open"),
                "closed": sum(1 for x in items_out if x["state"] == "closed"),
                "issues": items_out,
            }
        )
    return phases_payload


def _parse_deferred_backlog(
    bodies: Sequence[str], issues_by_num: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract open deferred backlog issues from the register tables.

    Args:
        bodies: Bodies of the child trackers of `#727`; a table row that
            starts with ``| #<n>`` registers issue ``n``. Rows are unioned.
        issues_by_num: All repository issues by number.

    Returns:
        Open backlog issues: those a register row names, plus those labelled
        ``deferred`` with ``upstream-bug`` or ``enhancement``.
    """
    nums_in_169 = {
        int(m.group(1))
        for body in bodies
        for line in body.splitlines()
        if line.startswith("| #") and (m := re.match(r"^\|\s*#(\d+)\b", line))
    }
    deferred_items: list[dict[str, Any]] = []
    for num, item in sorted(issues_by_num.items()):
        if str(item.get("state")) != "open":
            continue
        labels = {str(lb["name"]) for lb in item.get("labels", [])}
        if labels & {"tracking", "deferred-scope"}:
            continue
        is_backlog_label = "deferred" in labels and bool(
            labels & {"upstream-bug", "enhancement"}
        )
        if is_backlog_label or num in nums_in_169:
            kind = "enhancement" if "enhancement" in labels else "upstream-bug"
            deferred_items.append(
                {
                    "number": num,
                    "title": str(item["title"]),
                    "state": str(item["state"]),
                    "kind": kind,
                    "parity_pending": "parity-pending" in labels,
                    "needs_confirmation": "needs-confirmation" in labels,
                }
            )
    return deferred_items


def _parse_169_backlog(
    body_169: str, issues_by_num: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract open deferred backlog issues from one register body."""
    return _parse_deferred_backlog([body_169], issues_by_num)


def _sub_issues(repo: str, parent: int) -> list[dict[str, Any]]:
    """List the sub-issues of ``parent`` (GitHub's sub-issues API, paged)."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = _gh_api_json(
            f"repos/{repo}/issues/{parent}/sub_issues?per_page=100&page={page}"
        )
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return out


def sync_github_snapshot(
    snapshot_path: pathlib.Path, repo: str = "joshspeagle/FlexMESA"
) -> dict[str, Any]:
    """Query GitHub API for `#140`, the deferred registers and open PRs.

    The deferred register is the union of the tables in the bodies of every
    child tracker of `#727` (`#169`, `#728`, and later continuations).
    """
    issue_140 = _gh_api_json(f"repos/{repo}/issues/140")
    deferred_trackers = _sub_issues(repo, DEFERRED_SUPER_TRACKER)

    all_items: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = _gh_api_json(
            f"repos/{repo}/issues?state=all&per_page=100&page={page}"
        )
        if not isinstance(batch, list) or not batch:
            break
        all_items.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    issues_by_num: dict[int, dict[str, Any]] = {}
    open_prs: list[dict[str, Any]] = []
    for item in all_items:
        num = int(item["number"])
        if "pull_request" in item:
            if item.get("state") == "open":
                open_prs.append(
                    {
                        "number": num,
                        "title": str(item["title"]),
                        "draft": bool(item.get("draft", False)),
                    }
                )
        else:
            issues_by_num[num] = item

    payload = {
        "repo": repo,
        "tracker_issue": 140,
        "backlog_issue": 169,
        "phases_140": _parse_140_phases(
            str(issue_140.get("body", "")), issues_by_num
        ),
        "open_pull_requests": sorted(open_prs, key=lambda x: int(x["number"])),
        "deferred_169": _parse_deferred_backlog(
            [str(t.get("body") or "") for t in deferred_trackers],
            issues_by_num,
        ),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
