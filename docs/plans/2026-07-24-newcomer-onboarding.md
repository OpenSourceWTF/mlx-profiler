# Newcomer-ready profiler onboarding implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:executing-plans` to implement this plan task-by-task.

**Goal:** Give a newcomer one verified path from cloning the public profiler to
capturing, validating, viewing, and contributing traces, with safe recipes for
the complete preserved experiment lineage.

**Architecture:** `mlx-profiler` owns source build, census capture, experiment,
and profiler-contribution guidance. `metal-dispatch-viz` owns hosted/local
inspection and trace contribution guidance. Full cross-repository URLs join the
workflow while the historical experiment branch tips remain immutable.

**Tech Stack:** Markdown, CMake, Python/MLX, Node.js, React/Vite/Express, GitHub
Pages.

**Assumptions:**

- Assumes Apple silicon with full Xcode for the first build — the Metal
  profiler will not run on non-Metal hosts.
- Assumes experiment users explicitly choose the historical complete-tip branch
  — the recipes do not make capture-replay available on current `main`.
- Assumes the Cloudflare Worker secrets remain unavailable — the plan documents
  the LLM range API as pending and does not deploy it.

---

## File structure

### `OpenSourceWTF/mlx-profiler`

- `README.md`: first-screen fork identity and routing.
- `PROFILER.md`: canonical census build, capture, validation, and viewing
  tutorial.
- `EXPERIMENTS.md`: immutable branch ledger plus safe, runnable experiment
  recipes.
- `CONTRIBUTING.md`: fork-specific contribution and verification preface,
  preserving upstream rules.
- `docs/specs/2026-07-24-newcomer-onboarding-design.md`: approved design.
- `docs/plans/2026-07-24-newcomer-onboarding.md`: this execution plan.

### `OpenSourceWTF/metal-dispatch-viz`

- `README.md`: hosted application entrypoint, local-private-trace distinction,
  profiler capture link, and current LLM API status.
- `docs/submitting-traces.md`: canonical capture/build link and consistent
  terminology.

## Task 1: Repair the profiler entrypoint and census quickstart

**Files:**

- Modify: `mlx-profiler/README.md`
- Modify: `mlx-profiler/PROFILER.md`

**Security flag:** none

- [ ] Replace the upstream-first top navigation with profiler links and add an
  explicit warning that `pip install mlx` installs official MLX without the
  dispatch census.
- [ ] Add a linear quickstart to `PROFILER.md` with:
  `git clone`, `python3 -m venv`, build prerequisites, exact CMake configure and
  install commands, and a `mx.__file__` assertion rooted in the checkout.
- [ ] Add a copyable queued GPU workload, terminal-summary validator, hosted
  workbench URL, and local Express command.
- [ ] Retain the existing bounded-sink, schema, diagnostic cap, CPU contracts,
  CLT/prebuilt-metallib, and upstream-sync contracts without weakening them.
- [ ] Verify the rendered command order and reject stale instructions:

  ```bash
  git diff --check
  ! rg -n 'Drop a census JSONL file onto.*index\.html|pip install mlx$' \
    PROFILER.md
  rg -n 'git clone https://github.com/OpenSourceWTF/mlx-profiler.git|mx.__file__|final.*true|dropped_rows|https://mlx-profiler.opensource.wtf' \
    README.md PROFILER.md
  ```

## Task 2: Make preserved experiments runnable and contributions unambiguous

**Files:**

- Modify: `mlx-profiler/EXPERIMENTS.md`
- Modify: `mlx-profiler/CONTRIBUTING.md`

**Security flag:** none

- [ ] Add a complete-tip build from a clean clone, including the historical
  branch switch, CMake configuration, install, import-path proof, and opt-in
  environment gates.
- [ ] Add copyable recipes matching the published bindings for:
  `capture_compiled`, synchronous replay, asynchronous submit/wait,
  `make_feedback_plan`, partial writes, `write_input_range`,
  `make_chain_plan`, `chain_submit`, `chain_wait`,
  `MLX_CHAIN_GPU_TIMING`, and `MLX_CHAIN_GPU_TIMING_M2_SPLIT`.
- [ ] Place the `v0.31.2`, serial-use, one-in-flight, pinned-buffer,
  address-stability, and diagnostic-only limitations beside those recipes.
- [ ] Link the published `tests/s2b_alpha_toy.py` as the GPU correctness smoke.
- [ ] Add a fork-specific preface to `CONTRIBUTING.md` covering repository
  routing, immutable receipt branches, no force pushes, opt-in/no-sync
  instrumentation, CPU and Metal gates, and trace-submission routing.
- [ ] Verify documented APIs and gates exist on the complete tip:

  ```bash
  for symbol in capture_compiled make_feedback_plan replay_submit \
    replay_submit_partial write_input_range make_chain_plan chain_submit \
    chain_wait MLX_CHAIN_GPU_TIMING MLX_CHAIN_GPU_TIMING_M2_SPLIT; do
    git grep -q "$symbol" origin/experiments/s4-stage-split -- \
      python/src/metal.cpp mlx/backend/metal/capture_replay.cpp
  done
  git diff --check
  ```

## Task 3: Align the public workbench onboarding

**Files:**

- Modify: `metal-dispatch-viz/README.md`
- Modify: `metal-dispatch-viz/docs/submitting-traces.md`

**Security flag:** none

- [ ] Put `https://mlx-profiler.opensource.wtf` in the first-screen quickstart
  and explain that it contains curated public traces but cannot read a local
  folder.
- [ ] Route arbitrary/private traces to
  `npm start -- --trace-dir /absolute/path` and route capture to the profiler's
  canonical `PROFILER.md`.
- [ ] State that the LLM range endpoint remains unavailable until the
  Cloudflare Worker is deployed; do not publish a currently returning-404 URL
  as a usable API.
- [ ] Ensure the submission guide links directly to `PROFILER.md` and retains
  naming, terminal evidence, curation, privacy, and issue-template rules.
- [ ] Run the workbench gate:

  ```bash
  npm test
  npm run build
  npm run verify:pages
  npm audit
  git diff --check
  ```

## Task 4: Verify the complete newcomer journey and publish

**Files:**

- Verify all changed files in both repositories.
- Update GitHub repository metadata for `OpenSourceWTF/metal-dispatch-viz`.

**Security flag:** none

- [ ] Configure and run the profiler CPU contract suite from the newcomer
  worktree:

  ```bash
  cmake -S . -B build/census-cpu \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_TESTS=ON
  cmake --build build/census-cpu \
    --target dispatch_census_smoke dispatch_census_queue_smoke
  ctest --test-dir build/census-cpu --output-on-failure \
    -R '^dispatch_census_'
  ```

- [ ] Use a temporary fresh clone of each local worktree to validate every
  changed relative Markdown link and the documented first-screen commands
  without relying on untracked files.
- [ ] Confirm the hosted site returns HTTP 200, the registry contains five
  traces, and the latest Pages deployment is green.
- [ ] Scan changed documentation for private absolute paths, credential
  patterns, placeholders, and claims that the experiments are serving
  defaults.
- [ ] Commit each repository intentionally, push both
  `docs/newcomer-ready` branches, and open focused draft pull requests.
- [ ] Review the final diffs and GitHub checks, mark ready, merge the profiler
  PR first and the workbench PR second, then set the workbench homepage:

  ```bash
  gh repo edit OpenSourceWTF/metal-dispatch-viz \
    --homepage https://mlx-profiler.opensource.wtf
  ```

- [ ] Re-fetch both public `main` branches and prove the merged documentation,
  links, hosted site, registry, and Pages deployment match the intended
  newcomer workflow.
