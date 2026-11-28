---
name: Deferred (upstream bug or enhancement)
about: An upstream MESA bug or an improvement over upstream, recorded now and not implemented before v0.1.0
title: "[Deferred bug] or [Deferred enhancement] <short description>"
labels: deferred
---

<!--
Open as a sub-issue of the current deferred register tracker (listed in super-tracker #727; #728 since 2026-11-28) and add a row to the register table in that tracker's body. #169 and its continuations together form the register.
Add exactly one kind label: `upstream-bug` or `enhancement`.
Add `parity-pending` if FlexMESA code currently departs from upstream here.
Add `needs-confirmation` if the upstream lines were not read at tag 26.4.1.
-->

## Kind

<!-- "Upstream bug" (upstream contradicts its own documentation, comments, units or a neighbouring line) or "Enhancement" (upstream is self-consistent; a different choice would be better). -->

## Upstream location (MESA `26.4.1`)

<!-- `<module>/private/<file>.f90:<line>` for every line involved. -->

## Mechanism

<!-- What upstream does, quoting the lines, and why it is a bug or could be improved. -->

## Effect

<!-- Which outputs, controls or test cases are affected, and the size of the effect if known. -->

## FlexMESA today

<!-- Either "reproduces upstream" with the FlexMESA `file` and function, or "departs" with what the code does instead and the package issue that restores upstream behaviour. -->

## Proposed change after `v0.1.0`

<!-- The fix or improvement, and whether it should be reported to the MESA project. -->

## Found in

<!-- The package issue or audit finding during which this was found. -->
