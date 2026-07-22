// Copyright © 2026 Apple Inc.
//
// S2b capture-replay (alpha) — internal Metal-side recording hooks.
//
// This header declares the choke-point recording hooks that the Metal
// CommandEncoder calls while a capture window is open, plus the pipeline ->
// MTL::Function registry the Device populates so that captured compute
// pipelines can be re-created with supportIndirectCommandBuffers=YES for use
// inside an MTLIndirectCommandBuffer.
//
// The public, backend-agnostic surface (the CaptureReplay handle and
// capture_begin()) lives in mlx/backend/metal/metal.h so it can be bound from
// Python and stubbed by the no-metal build without pulling in Metal types.
// This header is Metal-aware and is only included by the Metal backend.
//
// Zero overhead when disabled: recording() is a cached bool && one relaxed
// atomic load, and every hook site guards on it. The env var MLX_CAPTURE_REPLAY
// (set to any non-empty value) *arms* capture — it makes the Device retain the
// MTL::Function for every kernel so ICB pipelines can be rebuilt later. Without
// it, capture() throws a clear error.

#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstdint>
#include <atomic>

namespace mlx::core::metal::capture {

namespace detail {
// Initialised at static-init from getenv("MLX_CAPTURE_REPLAY").
extern bool g_enabled;
// Open only during the single eval that a capture records.
extern std::atomic<bool> g_recording;
} // namespace detail

// True iff MLX_CAPTURE_REPLAY is set: the Device retains kernel functions so a
// capture can rebuild ICB-capable pipelines. Cheap, inlined.
inline bool enabled() {
  return detail::g_enabled;
}

// True iff a capture recording window is currently open. Cheap, inlined.
inline bool recording() {
  return detail::g_enabled &&
      detail::g_recording.load(std::memory_order_relaxed);
}

// --- Choke-point recording hooks (no-ops unless recording()) ---------------
// The order of calls per dispatch mirrors MLX encoding: set pipeline, bind
// buffers / setBytes / threadgroup memory, then dispatch. note_dispatch_*
// finalizes the in-progress command and starts a fresh one.
void note_pipeline(MTL::ComputePipelineState* pipeline);
void note_buffer_bind(const MTL::Buffer* buf, uint64_t offset, uint32_t index);
void note_set_bytes(const void* data, std::size_t nbytes, uint32_t index);
void note_threadgroup_mem(std::size_t length, uint32_t index);
void note_dispatch_threads(MTL::Size grid, MTL::Size group);
void note_dispatch_threadgroups(MTL::Size grid, MTL::Size group);

// --- Device pipeline -> function registry (populated only when enabled()) ---
// Called once per kernel at pipeline-state creation. Retains fn so a captured
// pipeline can be re-created with supportIndirectCommandBuffers=YES.
void register_kernel_function(
    MTL::ComputePipelineState* pipeline,
    MTL::Function* fn);

// Look up the retained function for a captured pipeline; nullptr if unknown.
MTL::Function* function_for_pipeline(MTL::ComputePipelineState* pipeline);

// --- Recording window control (called from the public capture API) ---------
void begin_recording();
void end_recording();

} // namespace mlx::core::metal::capture
