# SPDX-License-Identifier: LGPL-3.0-or-later
"""Declarative MESA `.defaults` and `*_columns.list` schema and namelist tools.

This subpackage parses upstream MESA `<pkg>/defaults/*.defaults` and
`*_columns.list` files into structured schemas and validates legacy Fortran
`&namelist` (`inlist`) configurations without generated Fortran I/O boilerplate.
"""

from __future__ import annotations

from flexmesa.schema import defaults, namelist

__all__: list[str] = ["defaults", "namelist"]
