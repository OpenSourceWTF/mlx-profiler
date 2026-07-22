# Public MLX Profiler Fork Design

**Status:** Approved approach; awaiting written-spec review  
**Date:** 2026-07-22

## Goal

Create `OpenSourceWTF/mlx-profiler` as a real public GitHub fork of
`ml-explore/mlx`, port the existing dispatch-census instrumentation onto the
current official `main`, and establish a low-risk path for periodically merging
future official MLX changes.

## Scope

- Create the fork inside the `OpenSourceWTF` organization and preserve GitHub's
  fork-parent relationship to `ml-explore/mlx`.
- Rename the organization fork to `mlx-profiler` if GitHub initially creates it
  as `OpenSourceWTF/mlx`.
- Create a separate local checkout at
  `/Users/davidtai/projects/OpenSourceWTF/mlx-profiler`.
- Configure `origin` as `OpenSourceWTF/mlx-profiler` and `upstream` as
  `ml-explore/mlx`.
- Port only these profiler commits, in order, onto the official current `main`:
  - `b500b8c`: dispatch census and command-buffer timeline.
  - `6aa5a61`: diagnostic `MLX_MAX_ACTIVE_TASKS` and cap-wait pricing.
  - `fa7bbe6`: CLT-only build through prebuilt metallib reuse.
- Preserve the profiler's disabled-by-default behavior and queued-lane safety.
- Add profiler-specific usage and upstream-maintenance documentation without
  replacing upstream MLX documentation.
- Build and run CPU-safe verification before publishing the patched branch.
- Push the verified profiler history to the public fork. This push is explicitly
  authorized by David's 2026-07-22 request.

## Non-goals

- Do not publish commits after `fa7bbe6` from the local `s3-serving` lineage.
  Capture-replay, feedback blits, partial writes, and chained serving submits
  remain outside this repository's initial profiler branch.
- Do not modify, rebase, push, or rename the existing local `mlx-fork`
  `s3-serving` branch.
- Do not include the next P1-P5 profiler development round in the repository
  bootstrap. That work begins only after the public baseline is established and
  verified.
- Do not add scheduled upstream-sync automation. Upstream changes must pass a
  reviewed merge and the profiler verification gates before publication.
- Do not alter the separate `OpenSourceWTF/metal-dispatch-viz` repository.

## Repository and Branch Architecture

`OpenSourceWTF/mlx-profiler` remains part of GitHub's `ml-explore/mlx` fork
network. Its default `main` branch contains current official MLX plus the
profiler-only patch stack. A separate local checkout avoids changing remotes or
branch state in the serving-critical `mlx-fork` checkout.

Local remotes:

```text
origin    https://github.com/OpenSourceWTF/mlx-profiler.git
upstream  https://github.com/ml-explore/mlx.git
```

The initial port uses an allowlisted, ordered `cherry-pick -x` of the three
profiler commits. The `-x` trailers preserve local provenance even if conflict
resolution produces new public commit hashes. No merge from `s3-serving` is
allowed.

Future official updates use:

```text
fetch upstream -> merge upstream/main into a sync branch -> build/test ->
review profiler hook diffs -> merge to main -> push origin
```

Published `main` is never rebased or force-pushed for routine synchronization.

## Profiler Contract

The initial public fork preserves the existing contract:

- `MLX_DISPATCH_CENSUS` unset: census instrumentation is disabled and behavior
  remains equivalent to official MLX.
- `MLX_DISPATCH_CENSUS=<path>`: the native MLX backend emits JSONL operation,
  command-buffer, wait, and summary records for diagnostic use.
- `MLX_MAX_ACTIVE_TASKS` remains diagnostics-only, with the official default of
  `10` unchanged when the environment variable is absent.
- Instrumentation must not add GPU synchronization to the queued execution lane.
- Build truth is the CMake install target; clangd-only diagnostics are not a
  release gate.

## Documentation

Add `PROFILER.md` with:

- the discovery-only purpose and disabled-in-serving rule;
- the build command using `MLX_METAL_PREBUILT_LIB`;
- environment variables and JSONL record types;
- the relationship to `OpenSourceWTF/metal-dispatch-viz`;
- the exact upstream synchronization procedure and verification gates.

Keep upstream `README.md` substantially unchanged to minimize recurring merge
conflicts. Point the GitHub repository description at `PROFILER.md`.

## Verification and Publication Gates

Before the first push:

1. Verify the GitHub repository is public, `isFork=true`, and its parent is
   exactly `ml-explore/mlx` after any rename.
2. Verify the public branch begins at the current official `main`, not the old
   `v0.31.2` base.
3. Verify the three allowlisted profiler changes are present and no commits or
   source paths unique to capture-replay/serving were imported.
4. Configure and run the established CLT-only CMake build with the prebuilt
   metallib, then run `cmake --build build/cr --target install`.
5. Import the newly installed `mlx` package and verify it resolves from the new
   checkout rather than a stale site-package.
6. Run a CPU-safe disabled/enabled configuration check and JSONL schema smoke
   test that requires no exclusive GPU window.
7. Confirm `git status` is clean and review the full upstream-to-profiler diff.
8. Push only after all preceding gates pass. Do not run an A3B GPU window as
   part of repository bootstrap.

## Error Handling

- If organization forking or rename is denied, stop without creating a
  standalone repository; a standalone copy would lose the required fork-parent
  relationship.
- If any profiler commit conflicts with current upstream, resolve it against
  the current target implementation and rerun the build and verification gates.
  Do not preserve old topology merely to reproduce the old patch.
- If a capture-replay or serving-only file appears in the port diff, abort the
  publication, remove only the newly ported commit, and reconstruct from the
  three-commit allowlist.
- If the disabled configuration changes runtime behavior or the CLT build fails,
  do not push.

## Failure-mode Assessment

1. **Accidental serving-code publication — critical.** The local diagnostic tip
   is a descendant of the profiler and includes ten later replay/serving commits.
   The ordered three-commit allowlist plus a forbidden-surface diff gate prevents
   a wholesale lineage merge.
2. **Stale-base transplant — critical.** Official MLX is 164 commits ahead of
   the profiler's `v0.31.2` base. The new branch starts at current official
   `main`; each profiler change is reconciled against current ownership,
   scheduling, build, and Metal-backend behavior.
3. **Fork identity lost during naming — critical.** Creating an unrelated
   `mlx-profiler` repository would defeat GitHub fork synchronization. The
   operation creates the real fork first, renames only within GitHub's fork
   network, and verifies the parent metadata before any profiler push.
4. **Future upstream merge regresses instrumentation — minor but recurring.**
   Upstream sync is intentionally reviewed and gated rather than scheduled or
   force-updated. `PROFILER.md` records the repeatable maintenance procedure.

## Migration Sequence

1. Create and verify the public GitHub fork and final repository name.
2. Clone it into the new `mlx-profiler` workspace and configure remotes.
3. Port and verify the three profiler commits on current official `main`.
4. Add profiler and maintenance documentation.
5. Review the complete diff, commit documentation, and push verified `main`.
6. Treat that public commit as the base for the separate P1-P5 development
   round.
