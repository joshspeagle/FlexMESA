# SPDX-License-Identifier: LGPL-3.0-or-later
"""GitHub `#140`, `#169`, and pull-request snapshot collector (`#174`)."""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import urllib.request
from typing import Any

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


def _parse_169_backlog(
    body_169: str, issues_by_num: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract open `#169` post-parity deferred backlog issues."""
    nums_in_169 = {
        int(m.group(1))
        for line in body_169.splitlines()
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


def sync_github_snapshot(
    snapshot_path: pathlib.Path, repo: str = "joshspeagle/FlexMESA"
) -> dict[str, Any]:
    """Query GitHub API for `#140`, `#169`, and open PRs into JSON."""
    issue_140 = _gh_api_json(f"repos/{repo}/issues/140")
    issue_169 = _gh_api_json(f"repos/{repo}/issues/169")

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
        "deferred_169": _parse_169_backlog(
            str(issue_169.get("body", "")), issues_by_num
        ),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload
