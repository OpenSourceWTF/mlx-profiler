// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mlx/api.h"
#include "mlx/array.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

// --- S2b capture-replay (alpha) --------------------------------------------
//
// EXPERIMENTAL. Records the compute dispatch stream of one eval of a compiled
// graph into an MTLIndirectCommandBuffer, then replays it with new input
// values by writing into the same (pinned) buffers and issuing a single
// executeCommandsInBuffer — bypassing the host command-encode cost that the
// S2a census showed dominates the per-cycle decode budget.
//
// The captured buffers (inputs, outputs and every intermediate touched by the
// stream) are pinned in the Metal allocator so their addresses are stable
// across replays. See mlx/backend/metal/capture_replay.cpp for the full list
// of alpha invariants that are NOT yet handled.
//
// Backend-agnostic (no Metal types) so it binds from Python and stubs cleanly
// in the no-metal build.

/** Arm/open a capture recording window on the current process.
 *
 * Requires the env var MLX_CAPTURE_REPLAY to be set (it makes the Device retain
 * kernel functions from process start). The next eval's dispatch stream is
 * recorded; call CaptureReplay::capture() afterwards to build the handle. */
MLX_API void capture_begin();

/** A replayable handle over a captured compute dispatch stream (alpha). */
class MLX_API CaptureReplay {
 public:
  struct Impl;

  ~CaptureReplay();
  CaptureReplay(const CaptureReplay&) = delete;
  CaptureReplay& operator=(const CaptureReplay&) = delete;

  /** Close the recording window opened by capture_begin() and build an ICB
   * from the dispatch stream recorded during the intervening eval.
   * `inputs` are the graph leaves whose buffer contents are rewritten on
   * replay; `outputs` are the result arrays. Both must already be evaluated. */
  static std::shared_ptr<CaptureReplay> capture(
      const std::vector<array>& inputs,
      const std::vector<array>& outputs);

  /** Write `new_inputs` into the pinned input buffers (address-stability
   * asserted) and submit one command buffer that executes the ICB. Returns a
   * fresh copy of each pinned output array. */
  std::vector<array> replay(const std::vector<array>& new_inputs);

  /** Number of compute commands baked into the ICB. */
  size_t num_commands() const;

  /** Number of those commands carrying a setBarrier() after deferred barrier
   * pruning. Equal to num_commands() for a fully serial / fully dependent
   * stream (e.g. the alpha toy chain). */
  size_t num_barriers() const;

  /** Longest run of consecutive barrier-free (concurrent) commands. */
  size_t largest_barrier_free_run() const;

  /** Barrier-cause diagnostics (decide reuse vs a genuine per-layer chain). */
  size_t num_raw_barriers() const; // barriers a read-after-write triggered
  size_t num_waw_barriers() const; // barriers a write-after-write triggered
  size_t max_buffer_write_count() const; // writes to the single hottest buffer
  size_t hottest_barrier_buffer_barriers() const; // barriers the worst buffer caused
  size_t renamable_writes() const; // full-def reuse writes that could be renamed
  size_t renamed_writes() const; // actually renamed (0 unless rename enabled)

  /** The captured input / output arrays (pinned buffers). */
  const std::vector<array>& inputs() const;
  const std::vector<array>& outputs() const;

 private:
  explicit CaptureReplay(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace mlx::core::metal
