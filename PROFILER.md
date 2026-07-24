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

## Enable a census

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

Drop a census JSONL file onto the
[`metal-dispatch-viz`](https://github.com/OpenSourceWTF/metal-dispatch-viz)
`index.html` page to inspect its command-buffer overlap timeline.

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
