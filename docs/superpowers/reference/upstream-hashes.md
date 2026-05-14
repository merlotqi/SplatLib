# Upstream Reference Hashes

This file pins the upstream reference repositories used for the `ai-superpowers` worktree analysis and implementation.

Date pinned: 2026-05-13

## Repositories

- `reference/engine`
  - Commit: `6df566a371dbd7b33de20a12ad5e05f2465b2a9b`
- `reference/supersplat`
  - Commit: `5b6b8eecec8c7ebf9bbf1c820cba8041a6c50c69`
- `reference/splat-transform`
  - Commit: `bebac611fb1653a701f7f2412d433e89df4f7bf4`

## Purpose

- Keep CLI and data-flow comparisons stable while translating reference behavior into C++.
- Avoid accidental drift if the upstream reference repositories advance during implementation.
