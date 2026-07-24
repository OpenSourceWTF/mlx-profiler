# MLX Metal profiler experiment lineage

This repository publishes the complete OpenSourceWTF MLX Metal profiler
history. The branches are evidence and development instruments, not supported
serving defaults.

`main` tracks current official MLX plus the hardened dispatch-census port. The
preserved experiment line starts from MLX `v0.31.2` and is not rebased onto
`main`: keeping its original commits makes benchmark receipts and source
comparisons reproducible. Port an experiment forward through a reviewed merge
or a fit-for-purpose rewrite; do not force-push these published branches.

> [!WARNING]
> Capture-replay, device feedback, chain replay, GPU timing, and stage splitting
> are historical diagnostic APIs. They are not installed by `pip install mlx`,
> are not present on this repository's `main`, and are not supported serving
> defaults. Use them on isolated workloads with one replay or chain in flight.

This guide intentionally lives on `main`, whose URL remains stable while a
local checkout is on a historical branch:
[`EXPERIMENTS.md`](https://github.com/OpenSourceWTF/mlx-profiler/blob/main/EXPERIMENTS.md).

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

## Build the complete experiment tip

The complete tip contains every API described below. Start from a clean clone
so an official MLX wheel or a current-`main` build cannot shadow it:

```bash
git clone https://github.com/OpenSourceWTF/mlx-profiler.git
cd mlx-profiler
git switch experiments/s4-stage-split

# Python 3.12 is the verified interpreter for this historical branch.
python3.12 -m venv .venv-experiments
source .venv-experiments/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy>=2'

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  python -m pip install -e .
```

`MLX_CAPTURE_REPLAY` changes Metal device construction and must be set before
Python imports MLX. Verify both the loaded extension and experimental surface:

```bash
MLX_CAPTURE_REPLAY=1 python - <<'PY'
from pathlib import Path
import mlx.core as mx

root = Path.cwd().resolve()
loaded = Path(mx.__file__).resolve()
assert loaded.is_relative_to(root), (
    f"wrong MLX import: {loaded}; expected a path below {root}"
)
for name in (
    "capture_compiled",
    "make_chain_plan",
    "chain_submit",
    "chain_wait",
):
    assert hasattr(mx.metal, name), f"missing experimental API: {name}"
print("experiment import:", loaded)
PY
```

The branch is based on MLX `v0.31.2`; it intentionally lacks later official MLX
fixes. Do not install it over an application environment that expects current
MLX.

## Capture and replay one compiled graph

Only fixed-shape compiled graphs whose captured pipelines and buffers satisfy
the ICB contracts are candidates. Capture pins graph buffers and retained
weights for the handle's lifetime. Invalid or unsupported graphs fail during
capture rather than falling back to ordinary execution.

```bash
MLX_CAPTURE_REPLAY=1 python - <<'PY'
import numpy as np
import mlx.core as mx

w = mx.array(np.eye(8, dtype=np.float32))
mx.eval(w)

@mx.compile
def graph(x):
    return mx.tanh(x @ w + 0.25)

x0 = mx.zeros((1, 8), dtype=mx.float32)
handle = mx.metal.capture_compiled(graph, x0)
assert handle.num_commands > 0

x1 = mx.ones((1, 8), dtype=mx.float32)
replayed = handle.replay([x1])[0]  # submit + wait + copied output
reference = graph(x1)
mx.eval(reference)
np.testing.assert_array_equal(np.array(replayed), np.array(reference))
print("captured commands:", handle.num_commands)
PY
```

For a larger correctness and timing smoke, run the published toy from the same
checkout:

```bash
MLX_CAPTURE_REPLAY=1 python tests/s2b_alpha_toy.py
```

That smoke drives a live Metal GPU and compares every replay bitwise with the
normal compiled graph. Run it outside production serving and benchmark gates.

## Separate submit, wait, and read

`replay_submit` commits without waiting. The returned ticket is single-use.
Wait before reading outputs or rewriting buffers for the next cycle:

```python
ticket = handle.replay_submit([x1])
handle.replay_wait(ticket)
output = handle.read_output(0)
```

Successive submits share pinned input and output buffers. The diagnostic
contract is serial use: one replay command buffer in flight unless the caller
has designed and verified external double-buffering.

## Device-side feedback and partial input updates

A feedback plan copies selected output leaves into equal-sized input leaves
after the captured compute encoder. The blit stays in the same command buffer,
so recurrent state can advance without a host read and rewrite:

```bash
MLX_CAPTURE_REPLAY=1 python - <<'PY'
import numpy as np
import mlx.core as mx

@mx.compile
def recurrent_step(token_delta, state):
    next_state = mx.tanh(state + token_delta)
    decision = mx.sum(next_state)
    return decision, next_state

delta0 = mx.zeros((8,), dtype=mx.float32)
state0 = mx.zeros((8,), dtype=mx.float32)
handle = mx.metal.capture_compiled(recurrent_step, delta0, state0)

# Output leaf 1 (next_state) -> input leaf 1 (state).
feedback = handle.make_feedback_plan([(1, 1)])
assert feedback.num_blits == 1

# Seed every input on the first cycle.
ticket = handle.replay_submit(
    [mx.ones((8,), dtype=mx.float32), state0],
    feedback,
)
handle.replay_wait(ticket)

# Later cycles rewrite only the varying token delta. The state input retains
# the preceding cycle's device-side feedback.
ticket = handle.replay_submit_partial(
    [0],
    [mx.full((8,), 0.5, dtype=mx.float32)],
    feedback,
)
handle.replay_wait(ticket)
print("decision:", float(handle.read_output(0)))
print("fed state:", np.array(handle.read_input(1)))
PY
```

`replay_submit_partial(indices, arrays, feedback_plan)` leaves every unlisted
input byte untouched. `write_input_range` goes one level smaller: it copies a
contiguous source array into one byte range but does not submit work.

```python
# Replace the first two float32 values of input leaf 0, then commit without
# rewriting any complete input leaf.
patch = mx.array([0.25, 0.75], dtype=mx.float32)
handle.write_input_range(0, 0, patch)
ticket = handle.replay_submit_partial([], [], feedback)
handle.replay_wait(ticket)
```

The byte range must fit the logical leaf, and feedback source/destination leaves
must have identical byte sizes. All buffers remain subject to the handle's
address-stability assertions.

## Chain captured stages into one command buffer

`make_chain_plan` validates an equal-sized cross-handle output-to-input handoff.
`chain_submit` emits the ordered ICB stages and fenced blits into one Metal
command buffer; `chain_wait` is the cycle's single decision wait.

```bash
MLX_CAPTURE_REPLAY=1 python - <<'PY'
import numpy as np
import mlx.core as mx

@mx.compile
def producer(x):
    return mx.tanh(x + 1)

@mx.compile
def consumer(y):
    return y * 2

x0 = mx.zeros((8,), dtype=mx.float32)
producer_handle = mx.metal.capture_compiled(producer, x0)
consumer_handle = mx.metal.capture_compiled(consumer, x0)
handoff = mx.metal.make_chain_plan(
    producer_handle, 0, consumer_handle, 0
)

# Full-leaf range writes change pinned input bytes without submitting a
# standalone replay.
x1 = mx.arange(8, dtype=mx.float32)
producer_handle.write_input_range(0, 0, x1)

ticket = mx.metal.chain_submit(
    [(producer_handle, None), (consumer_handle, None)],
    [handoff],
)
spans = mx.metal.chain_wait(ticket)
result = consumer_handle.read_output(0)

reference = consumer(producer(x1))
mx.eval(reference)
np.testing.assert_array_equal(np.array(result), np.array(reference))
print("stage spans:", spans)  # [] when GPU counter sampling is unavailable
PY
```

Each handle may also carry a same-handle feedback plan in its
`(handle, feedback_plan_or_None)` stage tuple. Handoffs must flow from an
earlier stage to a later stage. Chain use is serial: exactly one chained command
buffer in flight.

## GPU stage timing and within-stage splitting

GPU boundary timing is opt-in and diagnostic. It does not affect decode
correctness; when Metal counter sampling is unavailable, `chain_wait` returns an
empty span list.

Within-stage splitting divides one stage's ICB command range into fenced compute
groups. The spec is
`<zero-based-stage-index>:<ascending-command-split-points>`. Derive and validate
the exact split against the captured handle rather than copying a split from a
different graph. This complete example captures a multi-command second stage,
splits it in half, submits the chain, checks parity, and reports any available
GPU spans:

```bash
MLX_CAPTURE_REPLAY=1 python - <<'PY'
import os
import numpy as np
import mlx.core as mx

weights = [mx.eye(8, dtype=mx.float32) for _ in range(4)]
mx.eval(weights)

@mx.compile
def producer(x):
    return x + 1

@mx.compile
def measured_stage(y):
    h = y
    for weight in weights:
        h = mx.tanh(h @ weight + 0.125)
    return h

x0 = mx.zeros((1, 8), dtype=mx.float32)
producer_handle = mx.metal.capture_compiled(producer, x0)
stage_handle = mx.metal.capture_compiled(measured_stage, x0)
assert stage_handle.num_commands >= 2, (
    f"expected a multi-command stage, got {stage_handle.num_commands}"
)

handoff = mx.metal.make_chain_plan(
    producer_handle, 0, stage_handle, 0
)
stage_index = 1
cut = stage_handle.num_commands // 2
split_spec = f"{stage_index}:{cut}"
parsed = mx.metal.chain_parse_split_spec(
    split_spec, stage_handle.num_commands
)
assert parsed["valid"], parsed

# These are read on the first chain_submit. Set them only after deriving a
# valid split from this captured handle.
os.environ["MLX_CHAIN_GPU_TIMING"] = "1"
os.environ["MLX_CHAIN_GPU_TIMING_M2_SPLIT"] = split_spec

x1 = mx.arange(8, dtype=mx.float32).reshape(1, 8)
producer_handle.write_input_range(0, 0, x1)
ticket = mx.metal.chain_submit(
    [(producer_handle, None), (stage_handle, None)],
    [handoff],
)
spans = mx.metal.chain_wait(ticket)
result = stage_handle.read_output(0)

reference = measured_stage(producer(x1))
mx.eval(reference)
np.testing.assert_array_equal(np.array(result), np.array(reference))

if spans:
    measured_groups = [
        span["group_index"]
        for span in spans
        if span["stage_index"] == stage_index and not span["is_blit"]
    ]
    assert measured_groups == list(range(len(parsed["groups"]))), (
        measured_groups, parsed["groups"]
    )

print("split:", split_spec, parsed["groups"])
print("kernel order:", stage_handle.command_kernel_names)
print("GPU spans:", spans)  # [] when counter sampling is unavailable
PY
```

With timing available, each span has `stage_index`, `is_blit`, `duration_ns`,
`valid`, and `group_index`. A malformed or out-of-range split stays on the
unsplit path. Group boundaries and `command_kernel_names` describe ICB command
order; neither is a per-kernel timestamp.

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
