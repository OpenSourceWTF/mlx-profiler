# MLX Dispatch-Census Profiler

This repository is a public fork of
[`ml-explore/mlx`](https://github.com/ml-explore/mlx) for diagnostic Metal
instrumentation. Its dispatch census is a discovery tool: it is disabled by
default and must remain disarmed in serving and gated benchmark runs.

The preserved capture-replay, device-feedback, chained-cycle, GPU-timing, and
within-stage split experiments are intentionally kept off `main`. Their exact
branches, commits, opt-in gates, and base-version caveats are indexed in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

The profiler records how MLX encodes work, groups dispatches into Metal command
buffers, overlaps host encoding with GPU execution, and waits on allocator,
scheduler, and synchronization boundaries. It does not add GPU synchronization
to the queued execution lane.

## First census quickstart

### Requirements

- Apple silicon running a native `arm64` shell and Python;
- macOS 14 or newer;
- full Xcode 15 or newer, including the Metal compiler;
- Python 3.10 or newer; and
- Git plus CMake 3.25 or newer.

Confirm the process architecture and active developer directory:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
uname -m
xcrun --find metal
python3 -c 'import platform; assert platform.machine() == "arm64"'
```

`uname -m` must print `arm64`, and `xcrun --find metal` must resolve inside full
Xcode. If only Command Line Tools are installed, use the
[prebuilt-metallib workflow](#build-with-command-line-tools) instead.

### 1. Clone and build this fork

```bash
git clone https://github.com/OpenSourceWTF/mlx-profiler.git
cd mlx-profiler

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Do not replace the final command with `pip install mlx`: that installs the
official package without this profiler.

Prove Python loaded the extension from this checkout:

```bash
python - <<'PY'
from pathlib import Path
import mlx.core as mx

root = Path.cwd().resolve()
loaded = Path(mx.__file__).resolve()
assert loaded.is_relative_to(root), (
    f"wrong MLX import: {loaded}; expected a path below {root}"
)
print("profiler import:", loaded)
PY
```

Stop if this assertion fails. Fix the environment once rather than capturing a
trace from the wrong MLX installation.

### 2. Capture a queued GPU workload

`MLX_DISPATCH_CENSUS` must be present before Python imports MLX. The profiler
cannot attach to an already-running process or live GPU.

```bash
mkdir -p captures
TRACE="$PWD/captures/first-census.jsonl"

MLX_DISPATCH_CENSUS="$TRACE" python - <<'PY'
import mlx.core as mx

x = mx.arange(1024 * 1024, dtype=mx.float32)
for step in range(64):
    x = mx.tanh(x * 1.0001 + step * 0.0001)
    if (step + 1) % 8 == 0:
        mx.async_eval(x)
mx.eval(x)
print("result:", float(x[0]))
PY
```

Let the process exit normally so the asynchronous writer emits its terminal
summary.

### 3. Validate the evidence

```bash
tail -n 1 "$TRACE" | python -c '
import json, sys
row = json.load(sys.stdin)
assert row.get("record") == "summary", "final row is not a summary"
assert row.get("schema_version") == 1, "unsupported schema"
assert row.get("final") is True, "missing final:true terminal summary"
assert row.get("complete") is True, "capture is incomplete"
assert row.get("dropped_rows") == 0, "capture dropped rows"
assert row.get("ops_total", 0) > 0, "capture contains no dispatches"
assert row.get("cbs_total", 0) > 0, "capture contains no command buffers"
print("valid census:",
      row["ops_total"], "ops,",
      row["cbs_total"], "command buffers")
'
```

Any failed assertion invalidates the trace as evidence. Do not repair summary
fields by hand; recapture on a fast local volume.

### 4. Inspect the trace

The [hosted Metal Dispatch Workbench](https://mlx-profiler.opensource.wtf)
contains curated public traces. It deliberately does not upload or open private
local files.

To inspect your trace locally:

```bash
git clone https://github.com/OpenSourceWTF/metal-dispatch-viz.git
cd metal-dispatch-viz
npm ci
npm start -- --trace-dir /absolute/path/to/mlx-profiler/captures
```

Open `http://127.0.0.1:4173/`, choose `first-census.jsonl`, and use the launch
selector when the file contains multiple launch windows. The local Express
server is read-only and keeps the trace on your machine.

See the workbench's [schema and evidence
contract](https://github.com/OpenSourceWTF/metal-dispatch-viz/blob/main/schema.md)
before interpreting ordered dispatch placement as timing. Schema v1 has no
per-operation timestamps or tensor producer/consumer identities, so it cannot
establish an output critical path.

## Capture configuration and schema

Set `MLX_DISPATCH_CENSUS` before Python starts so the native library observes it
during initialization:

```bash
MLX_DISPATCH_CENSUS=/absolute/path/census.jsonl \
  python3 your_workload.py
```

The destination must be a regular file, not a pipe, socket, or device. The
asynchronous queue is bounded at 65,536 rows: if the filesystem cannot keep up,
later summaries report `dropped_rows` and set `complete:false` instead of
allowing profiler memory to grow without bound. Treat such a trace as invalid
evidence and rerun it on a faster local volume.

When the variable is absent or empty, the sink does not create an output file.
The disabled path performs only a cached flag check at each hook; it creates no
sink state or writer thread. When enabled, producers enqueue records to a
dedicated writer so filesystem I/O cannot block Metal completion handlers or
MLX scheduler and allocator locks.

Every row currently carries `"schema_version":1`. When enabled, the JSONL
contains these record types:

- `op`: kernel name, dispatch geometry, argument bytes, buffer binds, sequence,
  and command-buffer identity.
- `cb`: host encode interval, GPU execution interval, operation range, and
  operation count for one committed command buffer.
- `wait`: a timed allocator, scheduler, backpressure, or GPU synchronization
  wait. Waits below 250 ns remain in the summary totals without emitting an
  individual row.
- `summary`: total operations, command buffers, and counts and durations by
  wait bucket, plus trace completeness. Synchronization emits a numbered,
  best-effort `final:false` live snapshot; process teardown emits one coherent
  terminal `final:true` snapshot without forcing a GPU drain.

`MLX_MAX_ACTIVE_TASKS` optionally changes the asynchronous evaluation task cap
for diagnostics. Its upstream-compatible default remains `10`; valid values are
integers from `1` through `1000000`. Do not change this setting in production or
interpret deep-queue results without accounting for lifetime safety.

Use the
[`metal-dispatch-viz`](https://github.com/OpenSourceWTF/metal-dispatch-viz)
local trace-folder server to inspect private JSONL, or browse curated traces at
[`mlx-profiler.opensource.wtf`](https://mlx-profiler.opensource.wtf).

## Build with Command Line Tools

The Command Line Tools SDK does not ship Apple's Metal compiler. Build a
metallib from this exact source revision on a machine with full Xcode, then pass
that artifact and its generated adjacent `.manifest` to the CLT build. CMake
rejects missing manifests, artifact hash changes, kernel-source changes, and
Metal/SDK/build-setting mismatches.

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cmake -S . -B build/native-metallib \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cmake --build build/native-metallib --target mlx-metallib

# If the artifact is moved, move mlx.metallib.manifest beside it too.

cmake -S . -B build/cr \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/python/mlx" \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=ON \
  -DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY="$PWD/python/mlx" \
  -DMLX_METAL_PREBUILT_LIB="$PWD/build/native-metallib/mlx/backend/metal/kernels/mlx.metallib"

cmake --build build/cr --target install
PYTHONPATH="$PWD/python" python3 -c \
  'import mlx.core as mx; print(mx.__version__, mx.__file__)'
```

The final import path must resolve inside this checkout. The CMake install
target is the build gate; editor-only clangd diagnostics are not. The install
contains both `mlx.metallib` and its verified manifest.

## CPU contract tests

The disabled/enabled sink behavior, bounded queue, pipeline-name lifecycle,
baseline JSONL fields, and diagnostic cap parser can be checked without
reserving a GPU benchmark window:

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

## Synchronize official MLX

This checkout should have `origin` pointing at this repository and `upstream`
pointing at `https://github.com/ml-explore/mlx.git`. Update through a reviewed
merge, never by rebasing or force-pushing published `main`:

```bash
git fetch upstream
git switch -c sync/upstream-YYYY-MM-DD main
git merge --no-ff upstream/main
```

After resolving changes against current MLX ownership and APIs, rerun the CPU
contracts, exact-source metallib build, CLT install build, and local import
check. Review the full `upstream/main..HEAD` diff for serving or capture-replay
surfaces before merging the sync branch and pushing it.

The initial port's design and publication boundaries are recorded in
[`docs/specs/2026-07-22-public-mlx-profiler-fork-design.md`](docs/specs/2026-07-22-public-mlx-profiler-fork-design.md).
