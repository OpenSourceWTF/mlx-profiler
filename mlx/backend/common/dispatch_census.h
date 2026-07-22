// Copyright © 2026 Apple Inc.
//
// S2a dispatch-stream census (env-gated diagnostic sink).
//
// When MLX_DISPATCH_CENSUS names an output path, this module records, to a
// single JSONL file, three kinds of record for one process:
//
//   record="op"   one line per command encoded through the Metal
//                 CommandEncoder: kind, kernel name, setBytes usage, buffer
//                 binds, grid / threadgroup dims, and its command-buffer index.
//   record="cb"   one line per committed Metal command buffer: host encode
//                 start/end (mach-absolute ns) and GPU start/end (from
//                 MTLCommandBuffer GPUStartTime/GPUEndTime, same clock domain),
//                 first/last op seq and op count. This is what exposes the
//                 host-encode vs GPU-execute overlap (or the lack of it).
//   record="wait" one line per host stall: a named bucket (allocator lock /
//                 new-buffer syscall / scheduler backpressure / GPU sync) and
//                 the wait duration in ns, so the serialization point can be
//                 attributed.
//
// The sink lives in backend/common (always compiled, Metal-free) so the Metal
// backend, the allocator, and the core scheduler can all feed it without a
// layering violation and without breaking non-Metal builds.
//
// Minimal disabled-path overhead: enabled() reads one cached bool and every hook
// site guards on it. No sink state, output file, or writer thread is created.

#pragma once

#include <cstddef>
#include <cstdint>

namespace mlx::core::census {

// Plain grid / threadgroup dimensions (mirrors MTL::Size without the include).
struct Dim3 {
  uint64_t x;
  uint64_t y;
  uint64_t z;
};

// Per-CommandEncoder census state. MLX may interleave multiple streams on one
// host thread, so command-buffer ownership cannot be thread-local.
struct CommandBufferState {
  const void* pipeline = nullptr;
  uint32_t set_bytes_calls = 0;
  uint64_t set_bytes_total_bytes = 0;
  uint32_t buffer_binds = 0;
  bool started = false;
  uint64_t uid = 0;
  uint64_t encode_start_ns = 0;
  uint64_t first_op_seq = 0;
  uint64_t last_op_seq = 0;
  uint64_t op_count = 0;
};

namespace detail {
// Defined in dispatch_census.cpp; initialised at static-init from getenv.
extern bool g_enabled;
} // namespace detail

// True iff MLX_DISPATCH_CENSUS is set to a non-empty path. Cheap, inlined.
inline bool enabled() {
  return detail::g_enabled;
}

// Open and validate the configured sink before entering the measured lane.
// Throws once with a clear diagnostic if the configured path is unusable.
void initialize();

// Stop accepting records, wait for in-flight producers, emit one final summary,
// and flush. Idempotent; callback-visible storage remains valid until exit.
void finalize();

// Parse the diagnostics-only async-eval task cap. Invalid and absent values
// preserve MLX's upstream default of 10.
int parse_max_active_tasks(const char* value);

// Host clock in the mach-absolute nanosecond domain. This is the same domain as
// MTLCommandBuffer GPUStartTime()/GPUEndTime() (CACurrentMediaTime-based), so
// host encode times recorded with now_ns() are directly comparable to the GPU
// timestamps passed to note_cb_gpu_times().
uint64_t now_ns();

// --- Per-op (compute command) census -------------------------------------

// Register a compute-pipeline pointer -> kernel name (called once per kernel at
// pipeline-state creation) so note_dispatch can resolve the bound pipeline.
void register_kernel(const void* pipeline, const char* name);

// Per-dispatch accumulation. Command boundary: bind pipeline, set arguments
// (setBytes + buffers), then dispatch. note_dispatch emits the "op" line and
// resets the per-command accumulators.
void note_pipeline(CommandBufferState& state, const void* pipeline);
void note_set_bytes(CommandBufferState& state, std::size_t nbytes);
void note_buffer_bind(CommandBufferState& state);
void note_dispatch(
    CommandBufferState& state,
    const char* dispatch_kind,
    Dim3 grid,
    Dim3 threadgroup);

// --- Per-command-buffer timeline ------------------------------------------

// Finalize the command buffer that is about to be committed: record its host
// encode-end time, first/last op seq and op count; advance the command-buffer
// index used by subsequent "op" lines; and return the index just committed so
// the caller can attach GPU times from that buffer's completion handler.
uint64_t note_cb_encode_end(CommandBufferState& state);

// Attach GPU execution times (nanoseconds, mach-absolute domain: pass
// GPUStartTime()*1e9 and GPUEndTime()*1e9) to a previously finalized command
// buffer and emit its "cb" line. Safe to call from a completion-handler thread.
void note_cb_gpu_times(uint64_t cb_index, double gpu_start_ns, double gpu_end_ns);

// --- Host wait accounting --------------------------------------------------

// Record a host stall of wait_ns in a named bucket. Emits a "wait" line and
// accumulates a per-bucket total dumped at flush.
void note_wait(const char* bucket, uint64_t wait_ns);

// Emit a non-final summary snapshot and flush queued rows to the sink.
void flush();

} // namespace mlx::core::census
