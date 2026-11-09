# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared pytest configuration for FlexMESA test suites."""

from __future__ import annotations

import hypothesis

# Disable per-example timing deadlines so first-call JAX JIT compilation or
# parallel pytest-xdist worker initialization never triggers FlakyFailure.
hypothesis.settings.register_profile(
    "flexmesa_ci",
    deadline=None,
)
hypothesis.settings.load_profile("flexmesa_ci")
