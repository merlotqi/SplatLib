# Transform And Playcanvas LOD Design

**Date:** 2026-05-13

## Goal

Implement two related but intentionally separate deliverables in the `ai-superpowers` worktree:

1. Translate the existing TypeScript CLI behavior from `reference/splat-transform/src/cli/index.ts` into the C++ `transform/` tool without inventing new CLI behavior.
2. Add a new `playcanvasLOD/` executable that depends only on `SPLAT::splat`, reads a single input splat file, runs multi-level `simplifyGaussians`, assigns generated `lod` values, merges the results, and writes LOD output.

## Pinned Reference Baseline

All comparison and translation work in this spec uses the following pinned upstream baselines:

- `reference/supersplat` at `5b6b8eecec8c7ebf9bbf1c820cba8041a6c50c69`
- `reference/splat-transform` at `bebac611fb1653a701f7f2412d433e89df4f7bf4`

These hashes are also recorded in `docs/superpowers/reference/upstream-hashes.md`.

## Current State

### `transform/`

- `transform/main.cpp` already contains a partial shell for global flag parsing, input/output handling, and final write flow.
- `transform/process.h` defines only a subset of the TypeScript process actions.
- `transform/process.cpp` is effectively empty, so the action pipeline is not implemented.
- `transform/reader.cpp` and `transform/writer.cpp` provide tool-local read/write helpers, but they are not public API.

### `playcanvasLOD/`

- No directory or executable exists yet.
- Public `splat` headers already expose the main building blocks needed for a minimal implementation:
  - readers such as `readPly`, `readSog`, `readKsplat`, `readSplat`, `readSpz`, `readVoxel`, `readLcc`
  - `simplifyGaussians`
  - `combine`
  - `writeLod`

## Design Constraints

- The `transform/` CLI should be a translation of the reference TypeScript CLI, not a redesign.
- The new `playcanvasLOD/` tool must be an independent executable and depend only on `SPLAT::splat`.
- Documentation for this work lives in the `ai-superpowers` worktree.
- `playcanvasLOD/` accepts a simplified interface for level generation:
  - `--levels 100% 50% 25%`
- The `lod` values are auto-generated from the order of the percentages:
  - first entry -> `lod = 0`
  - second entry -> `lod = 1`
  - third entry -> `lod = 2`
- If the input file already contains a `lod` column, `playcanvasLOD/` ignores it and rebuilds the `lod` assignments from scratch.

## Reference Translation Scope

The C++ `transform/` tool should follow the same command model as the reference CLI:

- global options are parsed once
- positional file arguments build a sequence of file entries
- file-local options are attached to the most recent file entry as `processActions`
- all input files except the final positional file are treated as inputs
- the final positional file is treated as output
- input-side actions run before inputs are combined
- output-side actions run after non-environment inputs are combined

The immediate implementation priority is to make the action pipeline real and correct for the LOD generation flow. That means the first working subset must include:

- `param`
- `lod`
- `decimate`

Additional actions already modeled by the TypeScript CLI should be translated into C++ in the same style rather than introducing unrelated shortcuts.

## `transform/` Design

### CLI Parsing

`transform/main.cpp` should become the translation point for the reference TypeScript CLI structure:

- keep current global options where already present
- parse repeated file-local flags into `ProcessAction` entries
- keep the “arguments are interpreted relative to the most recent positional file” behavior
- validate values close to parse time so runtime processing can assume typed actions

For this implementation cycle, the parser must correctly support at minimum:

- `--params key=value,...`
- `--lod <n>`
- `--decimate <n|n%>`

If a file-local option appears before any positional file, parsing should fail with a clear error.

### Process Pipeline

`transform/process.h` and `transform/process.cpp` should mirror the reference process pipeline structure closely enough that the C++ code stays easy to compare against the pinned TypeScript implementation.

The C++ action model should be extended to include:

- `Decimate`

Execution semantics:

- `param`
  - not a geometric transform
  - collected during read or ignored during process execution if already consumed
- `lod`
  - ensure a `lod` column exists
  - fill all rows in the current table with the requested level value
- `decimate`
  - convert percentage or absolute target into a keep count
  - clamp to `[0, current_row_count]`
  - call `simplifyGaussians`

This gives `transform/` the same core chain needed for commands such as:

```bash
SplatTransform input.ply --decimate 100% --lod 0 --decimate 50% --lod 1 out/lod-meta.json
```

The exact CLI spellings should remain aligned with the translated C++ parser.

### Input And Output Data Flow

The output flow remains conceptually the same:

1. Read each input file into one or more `DataTable`s.
2. Reject unsupported non-Gaussian tables.
3. Run each input file’s `processActions`.
4. Split environment tables from non-environment tables.
5. Combine non-environment tables.
6. Run output file `processActions` on the combined table.
7. Optionally combine environment tables.
8. Write the final output.

This preserves compatibility with the existing writer behavior, including LOD output through `transform/writer.cpp`.

## `playcanvasLOD/` Design

### Responsibility

`playcanvasLOD/` is a focused utility for generating PlayCanvas-ready LOD output from a single source file. It does not aim to be a general transform pipeline.

### CLI

The new executable should support a compact interface:

```bash
PlaycanvasLOD [OPTIONS] <input-file> <output-lod-meta.json> --levels 100% 50% 25%
```

Required behavior:

- exactly one input file
- exactly one output target
- one or more `--levels` percentages
- level indices assigned automatically in argument order

Useful supporting options:

- `--overwrite`
- `--iterations`
- `--lod-chunk-count`
- `--lod-chunk-extent`

### Read Path

Because the tool must depend only on `SPLAT::splat`, it cannot reuse `transform/reader.cpp`.

Instead, `playcanvasLOD/` should provide a minimal local dispatcher based on input extension or filename pattern and call the public reader functions directly. The initial supported formats only need to cover the formats currently relevant to the repository and existing toolchain. Unsupported formats should fail with a direct message.

### LOD Generation

For each requested percentage:

1. Compute target row count from the original input row count.
2. Clamp the result into valid bounds.
3. Run `simplifyGaussians` on the original base table, not on the previous simplified output.
4. Ensure a `lod` column exists on the simplified result.
5. Fill the entire table with the auto-generated level index.

After all levels are built:

1. Combine all generated level tables into one merged `DataTable`.
2. Call `writeLod`.

This “all levels derive from the same original input” behavior keeps the meaning of `100% 50% 25%` straightforward and deterministic.

### Existing `lod` Column Handling

If the input table already has a `lod` column:

- ignore its contents
- overwrite or recreate the level assignments in each generated output table

This avoids mixed semantics between pre-authored LOD content and tool-generated content.

## File Layout

### Modified

- `transform/main.cpp`
- `transform/process.h`
- `transform/process.cpp`
- `transform/CMakeLists.txt`
- `CMakeLists.txt`

### Added

- `playcanvasLOD/CMakeLists.txt`
- `playcanvasLOD/main.cpp`
- optional small local helper file if format dispatch becomes too noisy in `main.cpp`
- `docs/superpowers/reference/upstream-hashes.md`

## Validation Strategy

### `transform/`

Verify the translated CLI can:

- parse file-local `--decimate` and `--lod`
- execute them in order
- produce valid LOD output through the existing writer path

### `playcanvasLOD/`

Verify the new tool can:

- read one supported file
- accept `--levels 100% 50% 25%`
- produce merged output with generated `lod = 0,1,2`
- write a valid `lod-meta.json` output tree

### Practical Verification

If no formal automated tests exist yet for these executables, at minimum run:

- a successful build for the enabled targets
- one end-to-end command for `transform/`
- one end-to-end command for `playcanvasLOD/`

The implementation should not be considered complete on build success alone.

## Risks And Mitigations

### Risk: CLI drift from TypeScript reference

Mitigation:

- keep the translation anchored to the pinned `index.ts`
- prefer matching parser behavior over inventing C++-specific shortcuts

### Risk: `playcanvasLOD/` silently supports fewer formats than `transform/`

Mitigation:

- document supported formats explicitly in help text
- fail loudly on unsupported inputs

### Risk: ambiguous level semantics

Mitigation:

- define `--levels` as percentages of the original input
- define auto-generated `lod` values by positional order
- define that existing input `lod` data is ignored
