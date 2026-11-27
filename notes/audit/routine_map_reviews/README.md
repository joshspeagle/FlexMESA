# Routine-map review records (`DECISIONS #79`)

A review record lets a pull request raise a file in `benchmarks/dashboard/port_manifest.toml` from `absent` to `partial` without changing that file's implementing Rust or Python files. The evidence is a sampled independent review of the file's `routine_map`. The exception covers no other status change. `ported_unverified` still requires a change to the implementing files, and the verified tiers still require evidence artifacts.

## What a record is

Each review is two committed files in this directory:

- `<name>.md`, the review text. It states the proposal reviewed (branch and commit), the frame (files, mapped routines and lines by level), the sampling method and seed, how each sampled mapping was graded, the error rates (unweighted and line-weighted), the worst findings with upstream and FlexMESA `file:line`, the corrections required before the raise, and what was not verified. It should include a table of every sampled item with its grade.
- `<name>.toml`, the machine-read record:

  | Key | Type | Meaning |
  | :-- | :-- | :-- |
  | `schema_version` | integer | `1` |
  | `lane` | string | Reviewing lane (`"A"`) |
  | `proposal` | string | Issue or branch of the reviewed mapping |
  | `reviewed_commit` | string | Commit of the reviewed mapping |
  | `review` | string | Path of the review text, under `notes/audit/` |
  | `seed` | integer | Seed of the sample |
  | `sample_full`, `sample_partial` | integers | Sampled mappings claimed `full` and `partial` |
  | `over_claims`, `under_claims` | integers | Sampled mappings whose level is wrong |
  | `wrong_target`, `not_ported` | integers | Sampled mappings naming the wrong code, or no port |
  | `files` | list of strings | Manifest `upstream_path` of every file raised on this review |

## What the gate checks

When a pull request raises an entry from `absent` to `partial` without changing any of its `flexmesa_files`, `src/flexmesa/testing/_port_manifest_verify.py` requires all of the following:

- The entry has a non-empty `routine_map`.
- Every entry of that `routine_map` passes `routine_gate` (`routine_key` and `fn_item`) outright, with no allow-list credit.
- The pull request adds or changes a record here that:
  - lists the entry under `files`;
  - carries the integer keys above, with a non-empty sample;
  - names a committed review text under `notes/audit/`.

The judgement in a review is the reviewer's own. Corrections the review finds are applied to the mapping before the raise. Any file where the sample found a wrong target is fixed before that file is raised. The independent audit can reopen a raise like any closed package (`DECISIONS #21`).
