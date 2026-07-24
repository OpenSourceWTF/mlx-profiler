# Contributing to MLX

## Contributing to the OpenSourceWTF profiler fork

This repository is a fork of
[`ml-explore/mlx`](https://github.com/ml-explore/mlx). Send general MLX
framework changes upstream unless they specifically implement or maintain the
OpenSourceWTF Metal diagnostics.

Profiler pull requests target
[`OpenSourceWTF/mlx-profiler`](https://github.com/OpenSourceWTF/mlx-profiler):

- `main` tracks current official MLX plus the opt-in dispatch census described
  in [`PROFILER.md`](PROFILER.md).
- The branches indexed in [`EXPERIMENTS.md`](EXPERIMENTS.md) are immutable
  benchmark and development evidence. Do not rebase, force-push, or add
  documentation commits to their published tips.
- Port historical work forward on a new feature branch from current `main`.
  Preserve the target path's ownership, shapes, buffer layout, compilation
  behavior, and workload constraints rather than transplanting an experiment
  by name.
- Every diagnostic remains off by default. Enabled census instrumentation must
  not add GPU synchronization or block scheduler, allocator, or completion
  locks on filesystem I/O.
- Keep validation, environment parsing, and eligibility decisions at
  construction or installation boundaries. Do not add per-token, per-cycle, or
  per-dispatch engagement counters to a measured serving path.

After any profiler source edit, run the CPU contracts:

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

Metal changes additionally require a successful install build and an import
that resolves inside the checkout. Follow the exact-source build in
[`PROFILER.md`](PROFILER.md#first-census-quickstart), then run the focused GPU
workload outside serving and gated benchmark runs. Capture-replay changes also
require the published toy or a stronger parity test on the affected graph.

Authentic JSONL evidence, trace metadata, and new public runs belong in
[`OpenSourceWTF/metal-dispatch-viz`](https://github.com/OpenSourceWTF/metal-dispatch-viz);
follow its
[`Submitting a profiler run`](https://github.com/OpenSourceWTF/metal-dispatch-viz/blob/main/docs/submitting-traces.md)
guide. Never commit raw private captures, prompts, generated text, model
weights, credentials, tokens, private filesystem paths, or customer data here.

The remaining guidance below is inherited from upstream MLX and applies to
source changes in this fork as well.

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo.
2. If you've added code that should be tested, add tests.
3. If a change is likely to impact efficiency, run some of the benchmarks before
   and after the change. Examples of benchmarks can be found in `benchmarks/python/`.
4. If you've changed APIs, update the documentation.
5. Every PR should have passing tests and at least one review.
6. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   This should install hooks for running `black` and `clang-format` to ensure
   consistent style for C++ and python code.

   You can also run the formatters manually as follows:

   ```shell
   clang-format -i file.cpp
   ```

   ```shell
   black file.py
   ```

   or run `pre-commit run --all-files` to check all files in the repo.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to MLX, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
