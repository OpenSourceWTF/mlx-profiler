# Newcomer-ready profiler onboarding design

**Status:** Approved for implementation

## Goal

Make the public OpenSourceWTF MLX profiler and Metal Dispatch Workbench usable
by a technically capable newcomer without campaign-specific context. A reader
must be able to:

1. distinguish the profiler fork from the official `mlx` package;
2. clone, build, and import the profiler from source on Apple silicon;
3. capture and validate a dispatch-census JSONL file;
4. open that trace in the hosted or local workbench;
5. understand the evidence limits before drawing conclusions;
6. locate, build, and exercise every preserved experiment family safely; and
7. contribute code or an authentic trace through the correct repository.

## Chosen approach

Use the two existing repositories as one documented workflow:

- `OpenSourceWTF/mlx-profiler` owns source build, capture, validation,
  experiment recipes, branch/version boundaries, and profiler contribution
  rules.
- `OpenSourceWTF/metal-dispatch-viz` owns hosted and local trace inspection,
  schema interpretation, curation, privacy review, and trace submission.

The profiler README will lead with the fork workflow instead of upstream MLX's
PyPI installation. `PROFILER.md` will be the canonical census tutorial.
`EXPERIMENTS.md` will remain the immutable lineage ledger while adding
copyable, minimal recipes grounded in the published Python bindings and smoke
test. The workbench README and submission guide will link back to those
canonical instructions and prominently expose the hosted application.

The preserved experiment branch tips remain unchanged. Their commit identities
are benchmark evidence. Stable instructions therefore live on `main` and use
full GitHub links that remain available while a local checkout is on a
historical branch.

## Alternatives considered

### README-only repair

This would correct the accidental `pip install mlx` path quickly, but it would
leave capture-replay, device feedback, chain replay, timing, and stage splitting
discoverable only through bindings and source tests. It does not satisfy the
goal.

### Separate documentation website

A dedicated site could provide richer navigation, but it would duplicate
versioned repository documentation and introduce another deployment that can
drift from both source repositories. The current documentation volume does not
justify it.

## Profiler documentation contract

### Root README

The first screen will:

- identify this as a source-built diagnostic fork;
- link directly to the census quickstart, hosted workbench, experiment guide,
  and contribution guide;
- warn that `pip install mlx` installs official MLX without this
  instrumentation; and
- retain upstream MLX product information below a clearly marked boundary.

### Census quickstart

`PROFILER.md` will provide one linear workflow:

1. verify native `arm64`, macOS, Python, CMake, and full Xcode;
2. clone the public fork and create a virtual environment;
3. configure and build the Python bindings and exact-source metallib;
4. prove the imported extension resolves inside the checkout;
5. run a small queued GPU workload with `MLX_DISPATCH_CENSUS` set before
   import;
6. validate the terminal summary and confirm no dropped rows;
7. open the trace at `https://mlx-profiler.opensource.wtf` or through the
   local Express trace-folder server; and
8. keep the census disabled in serving and gated benchmark runs.

The existing Command Line Tools plus prebuilt-metallib path remains as an
advanced build variant, not the first-time quickstart.

### Experiment recipes

`EXPERIMENTS.md` will distinguish three levels:

- **lineage:** exact branches and commits;
- **support boundary:** historical `v0.31.2`, diagnostic-only, serial use, one
  command buffer in flight where required, and no production-serving claim;
- **recipes:** build/import, basic capture/replay, async submit/wait, same-handle
  device feedback, partial and range writes, cross-handle chain replay, GPU
  stage timing, and within-stage split parsing/reporting.

Every recipe must use an API signature present in
`origin/experiments/s4-stage-split:python/src/metal.cpp`. The basic replay recipe
will also point to the published `tests/s2b_alpha_toy.py` correctness smoke.
Examples will not claim general graph capturability or silently fall back to
normal MLX execution.

### Contribution rules

The fork-specific preface in `CONTRIBUTING.md` will explain:

- profiler work targets `OpenSourceWTF/mlx-profiler`, while general framework
  work normally belongs upstream;
- `main` carries the current upstream-based census;
- preserved experiment branches are immutable evidence;
- ports forward use reviewed feature branches rather than rebases or force
  pushes;
- instrumentation remains opt-in and must not add synchronization;
- the required CPU contracts, Metal build/import proof, and focused GPU
  verification; and
- trace contributions belong in `metal-dispatch-viz`.

The upstream MLX contribution guidance remains intact below that preface.

## Workbench documentation contract

The workbench README will:

- put the hosted application beside the local quickstart;
- state that the hosted build contains curated showcase traces and does not
  accept local uploads;
- route private or arbitrary traces to the local read-only Express server;
- link capture instructions directly to the profiler's `PROFILER.md`;
- state that the planned LLM range API is not deployed until the Cloudflare
  Worker route exists, without advertising a currently returning-404 endpoint;
  and
- retain the complete control, evidence, build, and contribution reference.

The submission guide will use the same canonical profiler link and terminology.
The repository homepage metadata will point to the hosted workbench.

## Error handling and safety

- An import-path assertion prevents accidentally profiling with the official
  wheel.
- Terminal trace validation fails on a missing final summary, unsupported
  schema, incomplete capture, or dropped rows.
- Instructions never suggest attaching to a running GPU process; capture starts
  with a new environment-gated process.
- Historical experiments are explicitly separated from supported `main`.
- Examples preserve construction-time opt-in gates and never introduce hot-path
  validation or fallback behavior.
- No documentation asks contributors to publish prompts, generated text,
  credentials, private paths, or model weights.

## Verification strategy

### Static documentation

- `git diff --check` in both repositories;
- relative Markdown link validation for every changed document;
- search for stale `drop ... index.html`, profiler-directed `pip install mlx`,
  placeholder text, private paths, and secret patterns;
- verify every documented experiment API against the complete-tip binding; and
- verify every named public branch and cross-repository URL.

### Executable commands

- configure, build, and run the profiler CPU contract suite;
- build/install the profiler Python extension and assert its checkout-local
  import path;
- run the published capture-replay toy on a Metal-capable lane when the
  historical branch build is available;
- run the workbench's full test, build, Pages verifier, and dependency audit
  gates; and
- smoke the hosted page and five-entry registry.

Commands that require a live Metal device or full Xcode are identified as
hardware gates. A documentation change is not considered verified by static
lint alone when a matching build is available locally.

## Rollout

Each repository receives a focused documentation pull request from
`docs/newcomer-ready`. Merge the profiler documentation first so the workbench's
cross-link resolves immediately, then merge the workbench documentation and set
its GitHub homepage URL. Existing experiment tips, trace artifacts, profiler
schema, application behavior, and serving defaults do not change.

## Failure-mode review

1. **A newcomer imports the official wheel.** Critical. Resolved with an
   explicit warning and a mandatory `mx.__file__` checkout assertion before
   capture.
2. **Examples overstate experiment safety or work on the wrong MLX base.**
   Critical. Resolved by pinning the complete experiment branch, naming the
   `v0.31.2` base, using published signatures, and retaining serial-use and
   address-stability warnings beside the examples.
3. **The two repositories drift.** Minor but recurring. Reduced through one
   canonical owner per topic, reciprocal full URLs, link validation, and a
   single hosted-workbench origin.

## Non-goals

- Porting capture-replay experiments onto current `main`;
- enabling any profiler or experiment in serving;
- changing JSONL schema or workbench arithmetic;
- deploying the pending Cloudflare Worker without its repository secrets; or
- replacing upstream MLX documentation.
