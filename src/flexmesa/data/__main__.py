# SPDX-License-Identifier: LGPL-3.0-or-later
"""Module CLI entry point for ``python -m flexmesa.data``."""

from __future__ import annotations

from flexmesa.data import fetch

if __name__ == "__main__":
    raise SystemExit(fetch.main())
