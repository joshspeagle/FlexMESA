# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the dashboard GitHub collector and the super-trackers (`#174`).

GitHub caps a parent issue at 100 sub-issues, so the trackers sit under
super-trackers (`#725` Work, `#726` Parity, `#727` Deferred). The collector
reads the deferred register from every child tracker of `#727`. The tests
replace the GitHub API with a fake, so they run without a network.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from flexmesa.testing import _dashboard_github as gh


def _issue(
    num: int, labels: tuple[str, ...] = (), state: str = "open", body: str = ""
) -> dict[str, Any]:
    return {
        "number": num,
        "title": f"issue {num}",
        "state": state,
        "labels": [{"name": n} for n in labels],
        "body": body,
    }


def _row(num: int) -> str:
    return f"| #{num} | `x.f90:1` | desc |"


BUG = ("deferred", "upstream-bug")


def test_super_tracker_constants_name_the_three_super_trackers() -> None:
    """The super-tracker numbers live in three constants."""
    assert (
        gh.WORK_SUPER_TRACKER,
        gh.PARITY_SUPER_TRACKER,
        gh.DEFERRED_SUPER_TRACKER,
    ) == (725, 726, 727)


def test_backlog_unions_rows_of_every_child_register_body() -> None:
    """A row in any child tracker's table registers its issue."""
    issues = {
        n: _issue(n)
        for n in (10, 11, 12, 13)  # no deferred label: only rows count
    }
    out = gh._parse_deferred_backlog(
        [f"intro\n{_row(10)}\n{_row(11)}\n", f"## table\n{_row(12)}\n"], issues
    )
    assert [x["number"] for x in out] == [10, 11, 12]


def test_backlog_keeps_labelled_issues_and_skips_closed_and_trackers() -> None:
    """Counting rules of an issue counted before are unchanged."""
    issues = {
        1: _issue(1, BUG),
        2: _issue(2, ("deferred", "enhancement", "parity-pending")),
        3: _issue(3, BUG, state="closed"),
        4: _issue(4, (*BUG, "tracking")),
        5: _issue(5, (*BUG, "deferred-scope")),
        6: _issue(6, (*BUG, "needs-confirmation")),
    }
    out = gh._parse_deferred_backlog([""], issues)
    assert [x["number"] for x in out] == [1, 2, 6]
    by = {x["number"]: x for x in out}
    assert by[1]["kind"] == "upstream-bug" and not by[1]["parity_pending"]
    assert by[2]["kind"] == "enhancement" and by[2]["parity_pending"]
    assert by[6]["needs_confirmation"]


def test_parse_169_backlog_wrapper_matches_the_union_of_one_body() -> None:
    """The single-body entry point is the union over one body."""
    issues = {7: _issue(7), 8: _issue(8, BUG)}
    body = _row(7)
    assert gh._parse_169_backlog(body, issues) == gh._parse_deferred_backlog(
        [body], issues
    )


class _FakeGitHub:
    """Serves the endpoints the collector reads."""

    def __init__(self, children: list[dict[str, Any]], issues: list[Any]):
        self.children = children
        self.issues = issues
        self.calls: list[str] = []

    def __call__(self, endpoint: str) -> Any:
        self.calls.append(endpoint)
        if endpoint.endswith("/issues/140"):
            return _issue(140, ("tracking",), body="## Work packages\n")
        if "/issues/727/sub_issues" in endpoint:
            page = int(endpoint.rsplit("page=", 1)[1])
            return self.children[(page - 1) * 100 : page * 100]
        if "/issues?state=all" in endpoint:
            page = int(endpoint.rsplit("page=", 1)[1])
            return self.issues[(page - 1) * 100 : page * 100]
        raise AssertionError(endpoint)


def test_sync_reads_register_bodies_of_all_727_children(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot's deferred list is the union over #169 and #728."""
    fake = _FakeGitHub(
        children=[
            _issue(169, ("tracking",), body=_row(20)),
            _issue(728, ("tracking",), body=_row(21)),
        ],
        issues=[_issue(n) for n in (20, 21, 22)]
        + [
            {**_issue(30), "pull_request": {}, "draft": True},
        ],
    )
    monkeypatch.setattr(gh, "_gh_api_json", fake)
    snap = tmp_path / "s" / "github_snapshot.json"
    payload = gh.sync_github_snapshot(snap)
    assert [x["number"] for x in payload["deferred_169"]] == [20, 21]
    assert payload["backlog_issue"] == 169 and payload["tracker_issue"] == 140
    assert payload["open_pull_requests"] == [
        {"number": 30, "title": "issue 30", "draft": True}
    ]
    assert json.loads(snap.read_text(encoding="utf-8")) == payload
    assert any("/issues/727/sub_issues" in c for c in fake.calls)
    assert not any(c.endswith("/issues/169") for c in fake.calls)


def test_sub_issues_pages_past_one_hundred_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Children are read page by page until a short page."""
    kids = [_issue(1000 + i) for i in range(130)]
    fake = _FakeGitHub(children=kids, issues=[])
    monkeypatch.setattr(gh, "_gh_api_json", fake)
    assert len(gh._sub_issues("o/r", 727)) == 130
    assert len([c for c in fake.calls if "sub_issues" in c]) == 2


def test_a_child_without_a_body_contributes_no_rows(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null body (GitHub returns ``None``) is an empty register."""
    child = _issue(728, ("tracking",))
    child["body"] = None
    fake = _FakeGitHub(children=[child], issues=[_issue(5, BUG)])
    monkeypatch.setattr(gh, "_gh_api_json", fake)
    payload = gh.sync_github_snapshot(tmp_path / "s.json")
    assert [x["number"] for x in payload["deferred_169"]] == [5]
