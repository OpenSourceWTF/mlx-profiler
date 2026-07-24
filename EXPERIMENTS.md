# MLX Metal profiler experiment lineage

This repository publishes the complete OpenSourceWTF MLX Metal profiler
history. The branches are evidence and development instruments, not supported
serving defaults.

`main` tracks current official MLX plus the hardened dispatch-census port. The
preserved experiment line starts from MLX `v0.31.2` and is not rebased onto
`main`: keeping its original commits makes benchmark receipts and source
comparisons reproducible. Port an experiment forward through a reviewed merge
or a fit-for-purpose rewrite; do not force-push these published branches.

## Published branches

| Branch | Tip | Contents |
| --- | --- | --- |
| [`main`](https://github.com/OpenSourceWTF/mlx-profiler/tree/main) | moving | Current upstream-based MLX plus the bounded, source-verifiable dispatch census. |
| [`s2a-dispatch-census`](https://github.com/OpenSourceWTF/mlx-profiler/tree/s2a-dispatch-census) | [`fa7bbe6`](https://github.com/OpenSourceWTF/mlx-profiler/commit/fa7bbe640d3a378570efb2d566b5d120bd300c3c) | Original dispatch census, diagnostic active-task cap, and CLT/prebuilt-metallib build. |
| [`s2b-capture-alpha`](https://github.com/OpenSourceWTF/mlx-profiler/tree/s2b-capture-alpha) | [`57f5bd3`](https://github.com/OpenSourceWTF/mlx-profiler/commit/57f5bd3c73ab3df50f0ef2e5f70d3c0cd4090d51) | First compiled Metal dispatch-stream capture into an indirect command buffer and synchronous replay. |
| [`s2b-beta`](https://github.com/OpenSourceWTF/mlx-profiler/tree/s2b-beta) | [`5360ca2`](https://github.com/OpenSourceWTF/mlx-profiler/commit/5360ca2a1c41e2bc42348108b124f4e8aa73ece5) | Capturable A3B graph fixes, allocator lifetime safety, barrier experiments, optional buffer renaming, persistent residency, and asynchronous submit/wait/read. |
| [`s3-device-feedback`](https://github.com/OpenSourceWTF/mlx-profiler/tree/s3-device-feedback) | [`b1654e2`](https://github.com/OpenSourceWTF/mlx-profiler/commit/b1654e298ff3bc02eeaf6e4899b4ce97c464f692) | Device-side output-to-input feedback blits, selective I/O, offset-correct read-back, and byte-range input writes. |
| [`s4-chain-replay`](https://github.com/OpenSourceWTF/mlx-profiler/tree/s4-chain-replay) | [`d2866d2`](https://github.com/OpenSourceWTF/mlx-profiler/commit/d2866d251b28867b1eb60a50426cac4b279594f8) | Multiple capture-replay handles chained into one command buffer and one decision wait. |
| [`experiments/s4-stage-timing`](https://github.com/OpenSourceWTF/mlx-profiler/tree/experiments/s4-stage-timing) | [`f73109a`](https://github.com/OpenSourceWTF/mlx-profiler/commit/f73109a11999ce13d6c43ccb1c93595b34ab319a) | Per-stage GPU boundary timing with corrected CPU/GPU timestamp correlation. |
| [`experiments/s4-stage-split`](https://github.com/OpenSourceWTF/mlx-profiler/tree/experiments/s4-stage-split) | [`a220757`](https://github.com/OpenSourceWTF/mlx-profiler/commit/a220757b9e225d01f37034bf50f6117be2134bca) | Experimental split of one stage's ICB into fenced command groups for within-stage attribution. |
| [`s3-serving`](https://github.com/OpenSourceWTF/mlx-profiler/tree/s3-serving) | [`a220757`](https://github.com/OpenSourceWTF/mlx-profiler/commit/a220757b9e225d01f37034bf50f6117be2134bca) | Original complete-tip branch name retained for receipt compatibility; identical tip to `experiments/s4-stage-split`. |

Every commit from the census through the final stage-split experiment is
reachable from `s3-serving` and `experiments/s4-stage-split`.

## Complete commit ledger

| Phase | Commit | Change |
| --- | --- | --- |
| S2a | [`b500b8c`](https://github.com/OpenSourceWTF/mlx-profiler/commit/b500b8cb864dadf6d910b7bbadd941600b5cf54e) | Environment-gated Metal dispatch-stream census and command-buffer timeline. |
| S2a | [`6aa5a61`](https://github.com/OpenSourceWTF/mlx-profiler/commit/6aa5a61692bdada2d0e393dd794f1e1b787fb93a) | Environment-tunable `MLX_MAX_ACTIVE_TASKS` and cap-wait pricing. |
| S2a | [`fa7bbe6`](https://github.com/OpenSourceWTF/mlx-profiler/commit/fa7bbe640d3a378570efb2d566b5d120bd300c3c) | CLT-only build through verified prebuilt metallib reuse. |
| S2b alpha | [`57f5bd3`](https://github.com/OpenSourceWTF/mlx-profiler/commit/57f5bd3c73ab3df50f0ef2e5f70d3c0cd4090d51) | Experimental capture-replay of a compiled dispatch stream. |
| S2b beta | [`0ef3bb9`](https://github.com/OpenSourceWTF/mlx-profiler/commit/0ef3bb9c230c6243cb4bab0490bbe2b6c931a2f8) | Capturability fixes for the real A3B `m2_verify` graph. |
| S2b beta | [`67b88be`](https://github.com/OpenSourceWTF/mlx-profiler/commit/67b88be76940d12c9d9e23bd26395bad3db0886a) | Allocator teardown safety for pinned, still-owned buffers. |
| S2b beta | [`88d9be0`](https://github.com/OpenSourceWTF/mlx-profiler/commit/88d9be022de6419e5bb810bad684e490eb16f2e7) | Deferred barrier-pruning experiment for ICB replay. |
| S2b beta | [`11a3812`](https://github.com/OpenSourceWTF/mlx-profiler/commit/11a38127aa745dbaa56919a06e8d79d198194609) | Barrier hazard-chain diagnostics and gated buffer renaming. |
| S2b beta | [`5360ca2`](https://github.com/OpenSourceWTF/mlx-profiler/commit/5360ca2a1c41e2bc42348108b124f4e8aa73ece5) | Async replay submit/wait/read and persistent residency. |
| S3 | [`244048c`](https://github.com/OpenSourceWTF/mlx-profiler/commit/244048cc2c6cb5b371cb113ed222bfa4a52fabe8) | Device-side state-feedback blits and selective capture-replay I/O. |
| S3 | [`f3a78dd`](https://github.com/OpenSourceWTF/mlx-profiler/commit/f3a78dd4f21b7499fd771fa1062a0500a2851bee) | Offset-correct feedback blits and diagnostic read-back accessors. |
| S3 | [`b1654e2`](https://github.com/OpenSourceWTF/mlx-profiler/commit/b1654e298ff3bc02eeaf6e4899b4ce97c464f692) | Sub-leaf byte-range input writes for draft delta rewrites. |
| S4a | [`d2866d2`](https://github.com/OpenSourceWTF/mlx-profiler/commit/d2866d251b28867b1eb60a50426cac4b279594f8) | `chain_submit`, `chain_wait`, and `make_chain_plan`: one command buffer per decode cycle. |
| S4a W1 | [`b0e041b`](https://github.com/OpenSourceWTF/mlx-profiler/commit/b0e041b741bf1a5aed099ec3a80906a45f7d8cec) | Per-stage GPU boundary timestamps for chained replay. |
| S4a W1 | [`f73109a`](https://github.com/OpenSourceWTF/mlx-profiler/commit/f73109a11999ce13d6c43ccb1c93595b34ab319a) | Correlation-based GPU-tick-to-nanosecond conversion fix. |
| S4a W1b | [`a220757`](https://github.com/OpenSourceWTF/mlx-profiler/commit/a220757b9e225d01f37034bf50f6117be2134bca) | Fenced within-stage ICB group splitting and per-group kernel metadata. |

## Experimental Python surface

The complete tip adds these public `mlx.core.metal` entry points:

- `CaptureReplay`, `CaptureReplayFeedbackPlan`, and
  `CaptureReplayChainPlan`;
- `capture_compiled`;
- per-handle `replay`, `replay_submit`, `replay_submit_partial`,
  `replay_wait`, selective read-back, feedback-plan construction, and
  `write_input_range`;
- `make_chain_plan`, `chain_submit`, and `chain_wait`;
- `chain_ticks_to_ns` and `chain_parse_split_spec` for diagnostic parity and
  controller tests.

Build and import from the selected branch before using any surface:

```bash
git switch experiments/s4-stage-split

cmake --build build/cr --target install

PYTHONPATH="$PWD/python" python3 -c \
  'import mlx.core as mx; print(mx.__file__, mx.metal.capture_compiled)'
```

## Opt-in gates and limitations

- Dispatch census requires `MLX_DISPATCH_CENSUS=/absolute/path/trace.jsonl`.
- Capture-replay requires `MLX_CAPTURE_REPLAY` to be non-empty before importing
  MLX. Optional replay buffer renaming additionally requires
  `MLX_CAPTURE_REPLAY_RENAME`.
- GPU stage timing requires `MLX_CHAIN_GPU_TIMING`.
- Within-stage splitting additionally uses
  `MLX_CHAIN_GPU_TIMING_M2_SPLIT=<stage>:<split-indexes>`.
- All gates default off. Do not enable census instrumentation in a serving or
  gated benchmark run.
- Capture-replay and chained replay are experimental, serial-use diagnostic
  surfaces with pinned-buffer and address-stability constraints. Read the
  branch docstrings and tests before use.
- The experiment line is based on MLX `v0.31.2`; it does not include later
  official MLX fixes until someone ports and verifies it against a newer
  upstream base.

The default branch remains the supported place to synchronize official MLX and
develop the hardened dispatch profiler.
