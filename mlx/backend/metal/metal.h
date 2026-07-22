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

// --- S4a chained submit (forward declarations) -----------------------------
// A cross-handle blit plan and one stage of a chained submit; full definitions
// live after CaptureReplay (they reference it). chain_submit is a friend so it
// can reach each stage handle's Impl (queue / ICB / residency) while encoding
// them all into ONE command buffer.
struct ChainPlan;
struct ChainStage;
MLX_API uint64_t chain_submit(
    const std::vector<ChainStage>& stages,
    const std::vector<std::shared_ptr<ChainPlan>>& handoffs);

/** A replayable handle over a captured compute dispatch stream (alpha). */
class MLX_API CaptureReplay {
 public:
  struct Impl;

  // S4a: chain_submit encodes several handles' ICBs into one command buffer, so
  // it needs each handle's Impl (residency arena, ICB, num_commands). Additive —
  // the per-handle submit/replay APIs are unchanged.
  friend uint64_t chain_submit(
      const std::vector<ChainStage>& stages,
      const std::vector<std::shared_ptr<ChainPlan>>& handoffs);

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

  /** Sub-leaf range write (S3 draft delta-rewrite). Memcpy the bytes of `src`
   * into the pinned input leaf `index` at `byte_offset` WITHIN that leaf's data
   * (addressed through the leaf's array.data() = buffer base + array.offset(), so
   * a suballocated/viewed leaf's own nonzero offset is honored), leaving every
   * other byte of the leaf — and every other leaf — untouched (they persist the
   * previous submit / range write). `src` must be contiguous; its nbytes plus
   * `byte_offset` must fit within the leaf's logical nbytes. Address stability is
   * asserted (the captured buffer must not have moved). Does NOT commit — pair
   * range writes with a following replay_submit_partial (which writes the small
   * cycle-varying leaves and commits ONE command buffer). The draft reserved-
   * window K/V leaves change only in the 1-2 rows appended per cycle; rewriting
   * just those rows (per (batch,head) contiguous run) replaces the whole-window
   * memcpy replay_submit(_partial) would pay. Serial-use only (the draft chain is
   * submit->wait per cycle), same as replay_submit_partial. */
  void write_input_range(size_t index, size_t byte_offset, const array& src);

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

// --- S4a chained submit: one command buffer for the whole decode cycle -------
//
// EXPERIMENTAL (S4a). The serving cycle runs four captured streams — draft, m2
// verify, committed append, target sample — each today a SEPARATE
// replay_submit + replay_wait, so each pays its own host submit/wait round-trip
// (the §24 "remainder" the complete-capture milestone exposed). chain_submit
// collapses them: it encodes every stage handle's executeCommandsInBuffer (plus
// its OWN same-handle feedback blits) into ONE MTLCommandBuffer, with the
// cross-stream data handoffs blitted device-side BETWEEN the stages, and commits
// it with a SINGLE host submit; chain_wait is the SINGLE waitUntilCompleted. Host
// input rewrites (write_input_leaf / write_input_range) stay host-side per handle
// BEFORE the chain submit, exactly as today — chain_submit only encodes GPU work.
//
// Cross-handle handoff (ChainPlan): today the controller reads a stream's output
// to the host and writes it into the next stream's input (draft's verify_input →
// m2's ids leaf; m2's logits → target's input; committed hidden/token → append).
// A ChainPlan makes that a device-side fenced blit copying a SOURCE handle's
// output leaf into a DESTINATION handle's input leaf — the cross-handle
// generalization of FeedbackPlan (which is same-handle output→input only). Build
// it with make_chain_plan(); its src stage MUST be encoded before its dst stage.
//
// Fence / ordering: MLX buffers are MTLResourceHazardTrackingModeUntracked, so
// the encoder boundary does NOT order anything. chain_submit serializes every
// encoder (compute and blit) with a FRESH MTLFence per boundary — encoder k+1
// waits on encoder k's fence and updates its own. That total order trivially
// satisfies every data dependency because producers are always encoded before
// consumers: a stage's ICB, then its feedback + outgoing cross-handle blits, then
// the next stage's ICB (which waits on those blits). One chained command buffer
// in flight at a time (serial-use, the chained-decode cadence; the same
// discipline the per-handle feedback path documents).
//
// Residency: each stage's compute encoder explicitly useResources() its own
// handle's pinned arena / heaps / dupes / packed_args (the fork's per-call
// residency path), so the chain needs no per-queue residency set spanning
// handles (the amortized per-handle MTLResidencySet is bound to that handle's own
// queue, not the chain queue). Cross-handle blits name their src/dst directly, so
// they are auto-resident. Address stability is asserted for EVERY handle (feedback
// + handoff src/dst) BEFORE anything is encoded — a moved arena fails cleanly.

/** A validated cross-handle blit: copy output leaf `src_out_index` of `src` into
 * input leaf `dst_in_index` of `dst`, fenced between the two handles' ICBs inside
 * a chained command buffer. Build with make_chain_plan (which validates ranges +
 * equal byte size). Holds raw handle pointers (identity only, resolved to the
 * pinned buffers at chain_submit time re-asserting address stability); the caller
 * MUST keep both handles alive while any plan referencing them is submitted. */
struct ChainPlan {
  const CaptureReplay* src = nullptr; // source handle (blit source, P_out)
  size_t src_out_index = 0; // output leaf of src
  const CaptureReplay* dst = nullptr; // destination handle (blit dest, P_in)
  size_t dst_in_index = 0; // input leaf of dst
  size_t nbytes = 0; // validated equal byte size of both leaves
};

/** Validate and build a cross-handle ChainPlan. `src_out_index` must be in range
 * for `src`'s outputs, `dst_in_index` in range for `dst`'s inputs, and the two
 * leaves byte-size-equal (throws otherwise). */
MLX_API std::shared_ptr<ChainPlan> make_chain_plan(
    const std::shared_ptr<CaptureReplay>& src,
    size_t src_out_index,
    const std::shared_ptr<CaptureReplay>& dst,
    size_t dst_in_index);

/** One stage of a chained submit: a captured handle + an OPTIONAL same-handle
 * feedback plan (its OWN P_out→P_in blits, advancing that handle's decode state),
 * blitted right after the handle's ICB inside the chained command buffer. */
struct ChainStage {
  std::shared_ptr<CaptureReplay> handle;
  std::shared_ptr<CaptureReplay::FeedbackPlan> feedback_plan; // or null
};

/** Encode every stage's ICB (+ its feedback blits) and the cross-handle
 * `handoffs` into ONE command buffer in dependency order (fenced per boundary,
 * see above) and commit WITHOUT waiting; returns an opaque ticket. `stages` must
 * be non-empty, each handle distinct (a handle's ICB runs once per chain), and
 * every handoff's src/dst must be among the stages with the src stage BEFORE the
 * dst stage. Throws (nothing encoded) on any invalid stage/handoff or a moved
 * arena. Serial-use only: one chained command buffer in flight. */
MLX_API uint64_t chain_submit(
    const std::vector<ChainStage>& stages,
    const std::vector<std::shared_ptr<ChainPlan>>& handoffs = {});

/** Block until the chained command buffer for `ticket` completes (the SINGLE
 * per-cycle waitUntilCompleted). Throws on an unknown/already-waited ticket or a
 * command-buffer error. After it returns, read each stage handle's small decision
 * outputs via read_output(); the advancing state was fed on-device by the blits. */
MLX_API void chain_wait(uint64_t ticket);

} // namespace mlx::core::metal
