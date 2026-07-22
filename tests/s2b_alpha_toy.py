#!/usr/bin/env python3
# Copyright © 2026 Apple Inc.
#
# S2b capture-replay ALPHA smoke + correctness + host-cost test (toy graph).
#
# NOT part of the pytest suite and NOT run by the author — it drives Metal and
# must run inside a GPU window. It:
#   1. builds a toy *compiled* graph (~30-50 compute dispatches: a chain of
#      1024x1024 matmul + bias + activation layers with fixed weights),
#   2. captures its dispatch stream into an MTLIndirectCommandBuffer via
#      mlx.core.metal.capture_compiled  (mx.metal.* below, mx = mlx.core),
#   3. replays it 100x with a fresh random input each iteration,
#   4. asserts the replay output is BITWISE identical to a normal (eager)
#      eval of the same compiled graph each time,
#   5. times per-iteration wall cost of normal-eval vs replay.
#
# REQUIREMENT: run with MLX_CAPTURE_REPLAY set, e.g.
#   MLX_CAPTURE_REPLAY=capture python tests/s2b_alpha_toy.py
# Without it, capture_compiled raises (the Device must retain kernel functions
# from process start so ICB pipelines can be rebuilt).

import os
import sys
import time

import numpy as np

import mlx.core as mx

N = 1024          # matrix dim
LAYERS = 16       # ~2-3 dispatches/layer after compile fusion -> ~30-48
ITERS = 100
SEED = 0


def build_compiled():
    """A compiled chain of matmul + bias + activation with FIXED weights.

    Weights/biases are closed-over constants (not inputs), so the only input is
    x. capture pins the weight buffers; replay reuses them and only rewrites x.
    """
    rng = np.random.default_rng(SEED)
    ws = [
        mx.array((rng.standard_normal((N, N)) / np.sqrt(N)).astype(np.float32))
        for _ in range(LAYERS)
    ]
    bs = [
        mx.array(rng.standard_normal((N,)).astype(np.float32))
        for _ in range(LAYERS)
    ]
    mx.eval(ws, bs)

    @mx.compile
    def toy(x):
        h = x
        for w, b in zip(ws, bs):
            h = mx.tanh(h @ w + b)
        return h

    return toy


def to_bytes(a):
    return np.array(a, copy=False).tobytes()


def main():
    if not os.environ.get("MLX_CAPTURE_REPLAY"):
        print(
            "ERROR: set MLX_CAPTURE_REPLAY (e.g. =capture) before running.",
            file=sys.stderr,
        )
        return 2

    mx.random.seed(SEED)
    toy = build_compiled()

    x0 = mx.array(np.random.default_rng(1).standard_normal((N, N)).astype(np.float32))
    mx.eval(x0)

    # --- capture -----------------------------------------------------------
    handle = mx.metal.capture_compiled(toy, x0)
    print(f"[s2b-alpha] captured num_commands = {handle.num_commands}")
    assert handle.num_commands > 0, "no commands captured"

    # --- correctness: replay must be BITWISE == eager compiled eval --------
    rng = np.random.default_rng(1234)
    normal_times = []
    replay_times = []
    mismatches = 0

    for it in range(ITERS):
        xin_np = (rng.standard_normal((N, N)) / 4.0).astype(np.float32)
        xin = mx.array(xin_np)
        mx.eval(xin)

        # normal eager eval of the same compiled graph (reference)
        t0 = time.perf_counter()
        ref = toy(xin)
        mx.eval(ref)
        normal_times.append(time.perf_counter() - t0)

        # replay
        t0 = time.perf_counter()
        out = handle.replay([xin])
        mx.eval(out)  # host-backed already, but forces materialization
        replay_times.append(time.perf_counter() - t0)

        rb = to_bytes(ref)
        ob = to_bytes(out[0])
        if rb != ob:
            mismatches += 1
            if mismatches <= 3:
                r = np.array(ref, copy=False)
                o = np.array(out[0], copy=False)
                maxabs = float(np.max(np.abs(r - o)))
                print(
                    f"[s2b-alpha] MISMATCH iter={it} max_abs_diff={maxabs:.3e}"
                )

    # --- report ------------------------------------------------------------
    def stats(ts):
        arr = np.array(ts[1:])  # drop first (warmup)
        return arr.mean() * 1e3, arr.min() * 1e3

    nm, nmin = stats(normal_times)
    rm, rmin = stats(replay_times)
    print("[s2b-alpha] results:")
    print(f"  iterations              : {ITERS}")
    print(f"  bitwise mismatches      : {mismatches}")
    print(f"  normal eval  mean/min ms: {nm:.3f} / {nmin:.3f}")
    print(f"  replay       mean/min ms: {rm:.3f} / {rmin:.3f}")
    print(
        "  NOTE: wall time includes GPU execute; the S2 win is the eliminated "
        "host command-encode, best isolated with async cadence (future work)."
    )

    if mismatches != 0:
        print("[s2b-alpha] FAIL: replay not bitwise-identical.")
        return 1
    print("[s2b-alpha] PASS: replay bitwise-identical across all iterations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
