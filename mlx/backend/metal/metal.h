// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
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
   * fresh copy of each pinned output array. Compat = submit + wait + read. */
  std::vector<array> replay(const std::vector<array>& new_inputs);

  // --- S3 serving: device-side state feedback + selective I/O ---------------
  //
  // A feedback plan is a list of (output-leaf, input-leaf) blits appended to the
  // SAME command buffer AFTER the ICB (an on-device copy P_out -> P_in). It lets
  // serving keep the advancing decode state on the GPU: instead of copying the
  // full state out to the host and back in every cycle, the state outputs are
  // blitted straight into the pinned state inputs, so the next replay reads
  // advanced state with ZERO host bytes. Combined with replay_submit_partial
  // (which rewrites only the small cycle-varying leaves — token ids + offsets),
  // a decode cycle pays only a ~11-leaf host memcpy instead of the full 91-leaf
  // in / 182-leaf out copy the Phase-1 no-fork path paid.
  //
  // Blit ordering: MLX buffers are MTLResourceHazardTrackingModeUntracked, so
  // Metal does NOT auto-order a blit encoder after a compute encoder. Each
  // feedback submit therefore fences the ICB compute encoder (updateFence) and
  // waits on that fence in the blit encoder (waitForFence) — the blits are
  // guaranteed to observe the ICB's completed outputs. Feedback submits MUST be
  // serial (submit -> wait per cycle): the fence is reused across cycles and is
  // only safe with at most one feedback command buffer in flight. That is the
  // chained-decode cadence; the depth-2 same-value pipeline never uses feedback.
  struct FeedbackPlan {
    struct Blit {
      size_t out_index; // output-leaf index (source, P_out)
      size_t in_index; // input-leaf index (destination, P_in)
      size_t nbytes; // validated equal byte size of both leaves
    };
    std::vector<Blit> blits;
  };

  /** Validate and build a feedback plan. Each (out_index, in_index) pair must be
   * in range and the two leaves must be byte-size-equal. Throws on any invalid
   * pair. The plan holds indices (resolved to the pinned buffers at submit time,
   * re-asserting address stability), so it stays valid for the handle's life. */
  std::shared_ptr<FeedbackPlan> make_feedback_plan(
      const std::vector<std::pair<size_t, size_t>>& pairs) const;

  /** Async split of replay (S3 host-submit isolation / pipelining).
   * replay_submit writes inputs into the pinned buffers and commits ONE command
   * buffer WITHOUT waiting, returning an opaque ticket. replay_wait blocks on a
   * ticket's completion. read_outputs copies the current pinned outputs into
   * fresh host arrays (call after the producing submit has been waited).
   * At pipeline depth > 1 the caller must use same-value inputs OR external
   * double-buffering — successive submits share the pinned input buffers.
   *
   * A non-null feedback_plan appends its P_out -> P_in blits to the same command
   * buffer after the ICB (fenced, see FeedbackPlan). */
  uint64_t replay_submit(
      const std::vector<array>& new_inputs,
      const std::shared_ptr<FeedbackPlan>& feedback_plan = nullptr);

  /** Partial-rewrite submit: memcpy ONLY the input leaves named in `indices`
   * (paired with `arrays`) into their pinned buffers; every other pinned input
   * is left untouched (it persists whatever the previous submit / feedback blit
   * left there). Address stability is asserted on the touched leaves (and, when a
   * feedback_plan is given, on its destination inputs). Same ticket / wait
   * semantics as replay_submit. */
  uint64_t replay_submit_partial(
      const std::vector<size_t>& indices,
      const std::vector<array>& arrays,
      const std::shared_ptr<FeedbackPlan>& feedback_plan = nullptr);

  void replay_wait(uint64_t ticket);
  std::vector<array> read_outputs();

  /** Host copy of ONE output leaf (the selective-read counterpart of
   * read_outputs). Call after the producing replay has been waited. */
  array read_output(size_t index);

  /** Host copy of ONE input leaf, addressed through the array's data pointer
   * (respecting its offset). Returns the leaf's CURRENT logical contents,
   * including bytes a feedback blit wrote into it. Diagnostic use: read back the
   * fed-back state inputs to isolate blit correctness (were the inputs advanced
   * correctly?) from execution / reference. Call after the producing replay has
   * been waited. */
  array read_input(size_t index);

  /** Debug view of a leaf's backing buffer. `buffer_id` is the MTL::Buffer* as
   * an opaque integer (identity only — never dereferenced); `offset` is the
   * array's byte offset within that buffer (nonzero for a view / suballocated
   * leaf); `nbytes` is the leaf's logical byte size; `buffer_length` is the
   * buffer's allocation length; `heap` is true when the buffer is suballocated
   * from the shared MTL::Heap (< 256 B). Reveals which state leaves sit at a
   * nonzero offset — the case a naive offset-0 blit would mis-copy. Backend-
   * agnostic (no Metal types) so it binds from Python. */
  struct LeafBufferInfo {
    uint64_t buffer_id = 0;
    int64_t offset = 0;
    uint64_t nbytes = 0;
    uint64_t buffer_length = 0;
    bool heap = false;
  };
  LeafBufferInfo input_buffer_info(size_t index) const;
  LeafBufferInfo output_buffer_info(size_t index) const;

  /** Zero-copy mx.array views of the pinned output buffers at `indices`.
   * ALIASING CONTRACT: the returned arrays alias the pinned output buffers in
   * place — they reflect the most recently completed replay and are only valid
   * until the next replay submit overwrites those buffers. The caller MUST
   * consume or copy them before resubmitting. No host bytes are moved. */
  std::vector<array> output_arrays(const std::vector<size_t>& indices) const;

  /** Whether the amortized persistent-residency path is active (macOS 15+). */
  bool residency_set_active() const;

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
