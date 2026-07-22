// Copyright © 2026 Apple Inc.
//
// S2b capture-replay (alpha). See capture_replay.h (internal recording hooks)
// and metal.h (public CaptureReplay handle) for the contracts.
//
// What this does
// --------------
// During one eval of a compiled graph, the CommandEncoder choke points feed
// every compute dispatch here: pipeline, buffer binds (buffer + offset + index),
// setBytes payloads (copied into a packed args MTLBuffer at 256-B-aligned slots
// and re-bound as real buffers), threadgroup-memory lengths, and grid /
// threadgroup dims. capture() flattens that recorded stream (across all of
// MLX's internal command buffers) into ONE MTLIndirectCommandBuffer:
//   * pipelines are re-created with supportIndirectCommandBuffers=YES from the
//     same MTL::Function (host-side only; identical kernel code, identical
//     numerics);
//   * every setBytes value is baked once as an ICB kernel-buffer argument
//     (indirect commands cannot use setBytes);
//   * setBarrier() is set only where a true hazard exists — a command that
//     touches a buffer written since the previous barrier (deferred barrier
//     pruning). A barrier makes all prior commands complete before this one, so
//     it clears the accumulated write set; independent commands between barriers
//     run concurrently. (The alpha put a barrier on every command — fully
//     serial — which the real graph's many small kernels paid heavily for.)
// Every MTLBuffer the stream touches is pinned in the allocator so its address
// is stable; replay() writes new input values into the pinned input buffers and
// issues a single executeCommandsInBuffer.
//
// S2b-beta fixes (to capture the REAL A3B m2_verify graph, not just the toy).
// Each is additive and guarded so the alpha toy path is unchanged:
//   * Blocker 1 (input-boundary copies): a copy on an input leaf
//     (reshape/astype/broadcast) is captured natively — it is just another
//     recorded compute dispatch, and its SOURCE is the pinned, rewritten input.
//     capture() now also ASSERTS that every declared input leaf is actually
//     read by the recorded stream (its buffer is in the arena), so a leaf the
//     graph never touches on the GPU fails loudly instead of being silently
//     ignored on replay.
//   * Blocker 2 (heap-suballocated small buffers): MLX suballocates buffers
//     < 256 B from a shared MTL::Heap, and the dedicated replay queue does NOT
//     carry the allocator's residency set. capture() records the distinct heaps
//     backing the arena and replay() useHeap()s them, so A3B's tiny buffers
//     (input_ids, int32 offsets, scalars) are resident. The toy has no < 256 B
//     buffers, so this list is empty there.
//   * Blocker 3 (linked-function kernels): the pipeline registry now retains
//     each kernel's linked (private) functions too, and capture() rebuilds the
//     ICB pipeline with a matching MTL::LinkedFunctions set. Empty for plain
//     kernels (the toy), so no behaviour change there.
//   * Teardown safety (allocator): the real graph pins buffers that stay owned
//     by live arrays — model weights bound by matmul/qmm dispatches, and state
//     leaves co-owned by the route's cache. The allocator now defers a pinned
//     buffer's free only when free() is actually requested, and unpin() runs the
//     real free ONLY for those; a still-owned pinned buffer is just un-pinned.
//     Without this, ~Impl would recycle/release live weights at handle
//     destruction. The toy only pins buffers it exclusively owns → unchanged.
//   * Deferred barrier pruning (perf): setBarrier() is emitted only on a command
//     that touches a buffer written since the previous barrier, instead of on
//     every command. This lets the bulk of each layer's independent kernels run
//     concurrently; num_barriers()/largest_barrier_free_run() report the result.
//     The toy's fully dependent chain still gets ~a barrier per command.
//   * Async replay (S3 lever, perf): replay is split into replay_submit (encode
//     + commit, no wait -> ticket) and replay_wait (block on a ticket), with
//     read_outputs() moved out of the timed path and a persistent MTLResidencySet
//     on the replay queue amortizing the per-call useResources/useHeap host cost.
//     This isolates the host command-submit cost and enables pipelined replay;
//     the synchronous replay() delegates to submit+wait+read (toy unchanged).
//
// Invariants STILL NOT handled (these WILL break a graph):
//   * Allocator reuse across evals: pinning holds captured buffers, but any
//     graph whose per-cycle buffer set is not identical to the captured one
//     (dynamic shapes, KV-cache growth that reallocates, control flow) is out
//     of scope — the two compiled entries in the A3B lane are shape-stable, the
//     toy alpha graph is fixed.
//   * setBytes values are baked as CONSTANTS. Any per-cycle-varying value that
//     MLX passes via setBytes (rather than a tensor buffer) would be frozen at
//     capture. The R1/compiled A3B stack already makes cycle-varying values
//     (token ids, offsets) tensor-resident; shapes/strides are cycle-invariant
//     for the shape-stable compiled entry, so freezing them is correct.
//   * Donation INTO the captured region: prevented for inputs by holding the
//     input arrays live during the recorded eval (non-donatable); intermediates
//     donate among themselves harmlessly (all pinned). Donation of an input's
//     buffer to an output is not separately guarded.
//   * Non-compute work (blits, fences across MLX command buffers) is not
//     replayed; only compute dispatches are captured. The per-command barrier
//     substitutes for the cross-encoder fences.
//   * Single GPU stream, single capture at a time; replay must not run
//     concurrently with other GPU work touching the pinned buffers.

#include "mlx/backend/metal/capture_replay.h"
#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlx::core::metal {

namespace capture {

namespace detail {

static bool init_enabled() {
  const char* v = std::getenv("MLX_CAPTURE_REPLAY");
  return v != nullptr && v[0] != '\0';
}

bool g_enabled = init_enabled();
std::atomic<bool> g_recording{false};

} // namespace detail

namespace {

// --- pipeline -> retained MTL::Function registry --------------------------
// Retains the primary function and any linked (private) functions the pipeline
// was built from, so capture() can rebuild an ICB-capable pipeline with an
// identical MTL::LinkedFunctions set (blocker 3).
struct FnEntry {
  NS::SharedPtr<MTL::Function> fn;
  std::vector<NS::SharedPtr<MTL::Function>> linked;
};
std::mutex g_fn_mtx;
std::unordered_map<MTL::ComputePipelineState*, FnEntry> g_functions;

// --- recorded dispatch stream ---------------------------------------------
struct BufferBind {
  const MTL::Buffer* buf;
  uint64_t offset;
  uint32_t index;
};
struct BytesBind {
  uint32_t index;
  std::vector<uint8_t> data; // copied setBytes payload
};
struct TgMemBind {
  uint32_t index;
  uint32_t length;
};
struct DispatchRec {
  MTL::ComputePipelineState* pipeline = nullptr;
  std::vector<BufferBind> buffers;
  std::vector<BytesBind> bytes;
  std::vector<TgMemBind> tgmem;
  // Buffers this dispatch WRITES (from set_output_array/register_output_array
  // and set_buffer's dual in/out binds). Drives the deferred barrier-pruning
  // hazard rule; the read set is every bound buffer in `buffers`.
  std::unordered_set<const MTL::Buffer*> write_bufs;
  bool uses_threads = false; // true: dispatchThreads, false: dispatchThreadgroups
  MTL::Size grid{1, 1, 1};
  MTL::Size group{1, 1, 1};
};

struct Recorder {
  std::mutex mtx;
  std::vector<DispatchRec> dispatches;
  DispatchRec pending;
  std::unordered_set<const MTL::Buffer*> distinct_buffers;

  void reset() {
    dispatches.clear();
    pending = DispatchRec{};
    distinct_buffers.clear();
  }
};
Recorder g_rec;

} // namespace

void register_kernel_function(
    MTL::ComputePipelineState* pipeline,
    MTL::Function* fn,
    const std::vector<MTL::Function*>& linked_functions) {
  if (!detail::g_enabled || pipeline == nullptr || fn == nullptr) {
    return;
  }
  FnEntry entry;
  entry.fn = NS::RetainPtr(fn);
  entry.linked.reserve(linked_functions.size());
  for (auto* lf : linked_functions) {
    if (lf != nullptr) {
      entry.linked.push_back(NS::RetainPtr(lf));
    }
  }
  std::lock_guard<std::mutex> lk(g_fn_mtx);
  g_functions.emplace(pipeline, std::move(entry));
}

MTL::Function* function_for_pipeline(MTL::ComputePipelineState* pipeline) {
  std::lock_guard<std::mutex> lk(g_fn_mtx);
  auto it = g_functions.find(pipeline);
  return it == g_functions.end() ? nullptr : it->second.fn.get();
}

std::vector<MTL::Function*> linked_functions_for_pipeline(
    MTL::ComputePipelineState* pipeline) {
  std::lock_guard<std::mutex> lk(g_fn_mtx);
  auto it = g_functions.find(pipeline);
  std::vector<MTL::Function*> out;
  if (it != g_functions.end()) {
    out.reserve(it->second.linked.size());
    for (auto& lf : it->second.linked) {
      out.push_back(lf.get());
    }
  }
  return out;
}

void note_pipeline(MTL::ComputePipelineState* pipeline) {
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  g_rec.pending.pipeline = pipeline;
}

void note_buffer_bind(
    const MTL::Buffer* buf,
    uint64_t offset,
    uint32_t index) {
  // Pin immediately: the owning array is still alive here, but intermediates
  // are freed the moment the eval finishes — pinning now (before free) is what
  // keeps their addresses stable for replay.
  metal::allocator().pin(const_cast<MTL::Buffer*>(buf));
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  g_rec.pending.buffers.push_back(BufferBind{buf, offset, index});
  g_rec.distinct_buffers.insert(buf);
}

void note_output_bind(const MTL::Buffer* buf) {
  // Called from register_output_array (an output array bind) and set_buffer (a
  // dual in/out bind) to classify `buf` as WRITTEN by the pending dispatch.
  // Conservative: any buffer we are unsure about is a write, which only adds
  // barriers. The address is already pinned by the preceding note_buffer_bind.
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  g_rec.pending.write_bufs.insert(buf);
}

void note_set_bytes(const void* data, std::size_t nbytes, uint32_t index) {
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  BytesBind b;
  b.index = index;
  b.data.resize(nbytes);
  if (nbytes > 0 && data != nullptr) {
    std::memcpy(b.data.data(), data, nbytes);
  }
  g_rec.pending.bytes.push_back(std::move(b));
}

void note_threadgroup_mem(std::size_t length, uint32_t index) {
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  g_rec.pending.tgmem.push_back(
      TgMemBind{index, static_cast<uint32_t>(length)});
}

void note_dispatch_threads(MTL::Size grid, MTL::Size group) {
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  g_rec.pending.uses_threads = true;
  g_rec.pending.grid = grid;
  g_rec.pending.group = group;
  g_rec.dispatches.push_back(std::move(g_rec.pending));
  g_rec.pending = DispatchRec{};
}

void note_dispatch_threadgroups(MTL::Size grid, MTL::Size group) {
  std::lock_guard<std::mutex> lk(g_rec.mtx);
  g_rec.pending.uses_threads = false;
  g_rec.pending.grid = grid;
  g_rec.pending.group = group;
  g_rec.dispatches.push_back(std::move(g_rec.pending));
  g_rec.pending = DispatchRec{};
}

void begin_recording() {
  {
    std::lock_guard<std::mutex> lk(g_rec.mtx);
    g_rec.reset();
  }
  detail::g_recording.store(true, std::memory_order_release);
}

void end_recording() {
  detail::g_recording.store(false, std::memory_order_release);
}

} // namespace capture

// ===========================================================================
// CaptureReplay handle
// ===========================================================================

namespace {

constexpr size_t kArgAlign = 256; // ICB kernel-buffer offset alignment

size_t align_up(size_t v, size_t a) {
  return (v + a - 1) / a * a;
}

} // namespace

struct CaptureReplay::Impl {
  NS::SharedPtr<MTL::CommandQueue> queue;
  NS::SharedPtr<MTL::IndirectCommandBuffer> icb;
  NS::SharedPtr<MTL::Buffer> packed_args;
  std::vector<NS::SharedPtr<MTL::ComputePipelineState>> icb_pipelines;

  size_t num_commands = 0;

  // Deferred barrier-pruning stats (see capture()): how many of the commands
  // carry a setBarrier(), and the longest run of consecutive barrier-free
  // (concurrent) commands. num_barriers == num_commands means fully serial
  // (the alpha behaviour / a fully dependent chain like the toy).
  size_t num_barriers = 0;
  size_t largest_barrier_free_run = 0;
  // Barrier-cause diagnostics — decide reuse (renamable, fix a) vs a genuine
  // per-layer chain (fix b). raw = a barrier a read triggered; waw = a barrier a
  // write triggered. max_buffer_write_count = writes to the single hottest
  // buffer (a shared scratch written by every layer shows up here).
  // hottest_barrier_buffer_barriers = how many barriers the single worst
  // offending buffer triggered. renamable_writes = full-def writes that reuse a
  // previously-written buffer and could be renamed to break a false chain;
  // renamed_writes = how many were actually renamed (0 unless rename enabled).
  size_t num_raw_barriers = 0;
  size_t num_waw_barriers = 0;
  size_t max_buffer_write_count = 0;
  size_t hottest_barrier_buffer_barriers = 0;
  size_t renamable_writes = 0;
  size_t renamed_writes = 0;

  // Renaming duplicates (fix a, gated by MLX_CAPTURE_REPLAY_RENAME). Owned here;
  // made resident on replay via useResources alongside the arena.
  std::vector<NS::SharedPtr<MTL::Buffer>> rename_dupes;

  // Async replay (S3 lever). Persistent residency set attached to the replay
  // queue amortizes the per-call useResources/useHeap host cost (macOS 15+ /
  // Metal3); use_residency records whether it is active (else replay_submit
  // falls back to per-call residency). inflight holds submitted-but-unwaited
  // command buffers keyed by an opaque ticket. Cached arena Resource* array for
  // the fallback path. Guarded by submit_mtx_ for pipelined use.
  NS::SharedPtr<MTL::ResidencySet> residency;
  bool use_residency = false;
  std::vector<const MTL::Resource*> arena_resources;
  std::unordered_map<uint64_t, NS::SharedPtr<MTL::CommandBuffer>> inflight;
  uint64_t next_ticket = 1;
  std::mutex submit_mtx;

  // Distinct pinned buffers (arena) — used for useResources and, at teardown,
  // for the deferred allocator free.
  std::vector<MTL::Buffer*> arena;

  // Distinct MTL heaps backing arena sub-allocations. MLX suballocates buffers
  // smaller than 256 B from a shared heap; the replay queue does NOT carry the
  // allocator's residency set (which wires the heap), so those small buffers
  // are only made resident if we useHeap() them explicitly (blocker 2). The
  // alpha toy has no <256 B buffers, so this list is empty there.
  std::vector<MTL::Heap*> heaps;

  // Captured graph leaves / results. inputs are rewritten on replay; outputs
  // are copied out. input_bufs mirrors the input buffers for the
  // address-stability assert.
  std::vector<array> inputs;
  std::vector<array> outputs;
  std::vector<const MTL::Buffer*> input_bufs;

  ~Impl() {
    // Drain any submitted-but-unwaited command buffers before touching the
    // buffers they read/write (they reference the pinned arena / dupes).
    for (auto& kv : inflight) {
      if (kv.second) {
        kv.second->waitUntilCompleted();
      }
    }
    inflight.clear();
    // Drop MTL objects first.
    residency.reset();
    icb.reset();
    packed_args.reset();
    icb_pipelines.clear();
    queue.reset();
    // Destroy the captured arrays BEFORE unpinning: their Data destructors call
    // allocator.free() on the pinned buffers, which is deferred (recorded) while
    // still pinned. Then unpin() runs the real free for exactly the buffers
    // whose free was deferred (the ones the capture was the last owner of —
    // intermediates and the input/output leaves); buffers still owned elsewhere
    // (model weights, state leaves co-owned by the route) are simply un-pinned,
    // never freed here. See allocator.cpp free()/unpin() and deferred_free_.
    inputs.clear();
    outputs.clear();
    input_bufs.clear();
    auto& alloc = metal::allocator();
    for (auto* b : arena) {
      alloc.unpin(b);
    }
    arena.clear();
  }
};

CaptureReplay::CaptureReplay(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CaptureReplay::~CaptureReplay() = default;

size_t CaptureReplay::num_commands() const {
  return impl_->num_commands;
}

size_t CaptureReplay::num_barriers() const {
  return impl_->num_barriers;
}

size_t CaptureReplay::largest_barrier_free_run() const {
  return impl_->largest_barrier_free_run;
}

size_t CaptureReplay::num_raw_barriers() const {
  return impl_->num_raw_barriers;
}

size_t CaptureReplay::num_waw_barriers() const {
  return impl_->num_waw_barriers;
}

size_t CaptureReplay::max_buffer_write_count() const {
  return impl_->max_buffer_write_count;
}

size_t CaptureReplay::hottest_barrier_buffer_barriers() const {
  return impl_->hottest_barrier_buffer_barriers;
}

size_t CaptureReplay::renamable_writes() const {
  return impl_->renamable_writes;
}

size_t CaptureReplay::renamed_writes() const {
  return impl_->renamed_writes;
}

const std::vector<array>& CaptureReplay::inputs() const {
  return impl_->inputs;
}

const std::vector<array>& CaptureReplay::outputs() const {
  return impl_->outputs;
}

void capture_begin() {
  if (!capture::enabled()) {
    throw std::runtime_error(
        "[capture_begin] MLX_CAPTURE_REPLAY is not set. Set it (e.g. "
        "MLX_CAPTURE_REPLAY=capture) before importing mlx so the Device "
        "retains kernel functions for ICB pipeline re-creation.");
  }
  capture::begin_recording();
}

std::shared_ptr<CaptureReplay> CaptureReplay::capture(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  // Close the recording window and take ownership of the recorded stream.
  capture::end_recording();

  std::vector<capture::DispatchRec> dispatches;
  std::vector<MTL::Buffer*> arena;
  {
    std::lock_guard<std::mutex> lk(capture::g_rec.mtx);
    dispatches = std::move(capture::g_rec.dispatches);
    arena.reserve(capture::g_rec.distinct_buffers.size());
    for (auto* b : capture::g_rec.distinct_buffers) {
      arena.push_back(const_cast<MTL::Buffer*>(b));
    }
    capture::g_rec.reset();
  }

  auto fail = [&arena](const std::string& msg) {
    // Unpin anything we pinned so a failed capture does not leak.
    auto& alloc = metal::allocator();
    for (auto* b : arena) {
      alloc.unpin(b);
    }
    throw std::runtime_error(msg);
  };

  if (dispatches.empty()) {
    fail(
        "[CaptureReplay::capture] No compute dispatches were recorded. Call "
        "capture_begin() before evaluating the graph, and make sure the eval "
        "actually ran on the GPU stream.");
  }

  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(mlx::core::Device::gpu);
  auto* mtl_device = d.mtl_device();

  auto impl = std::make_unique<Impl>();
  impl->arena = arena;
  impl->num_commands = dispatches.size();

  // Blocker 2: gather the distinct heaps backing any sub-allocated arena buffer
  // so replay can wire them explicitly (the replay queue has no residency set).
  {
    std::unordered_set<MTL::Heap*> heap_set;
    for (auto* b : impl->arena) {
      if (auto* h = b->heap()) {
        heap_set.insert(h);
      }
    }
    impl->heaps.assign(heap_set.begin(), heap_set.end());
  }

  // Blocker 1 (input-boundary copies): a copy inserted on an input leaf
  // (reshape/astype/broadcast) is captured NATIVELY — it is just another
  // compute dispatch in this backend, so it is recorded into the ICB, and its
  // SOURCE (the leaf we pin and rewrite on replay) is bound and pinned by
  // note_buffer_bind exactly like any other input. The address-stability assert
  // in replay() then guards that same source buffer. The only case that stays
  // unrecoverable is a leaf the compiled graph never reads on the GPU stream
  // (materialized outside the recorded eval): replay would memcpy into a buffer
  // no command reads. Detect that here and fail clearly — never hand back a
  // handle that silently drops an input.
  {
    std::unordered_set<const MTL::Buffer*> arena_set(
        impl->arena.begin(), impl->arena.end());
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto* ib = static_cast<const MTL::Buffer*>(inputs[i].buffer().ptr());
      if (ib == nullptr || arena_set.find(ib) == arena_set.end()) {
        fail(
            "[CaptureReplay::capture] Input leaf " + std::to_string(i) +
            " is not read by any captured dispatch: its buffer is absent from "
            "the recorded stream. The compiled graph consumed it only through a "
            "value materialized outside the recorded eval, so replay could not "
            "rewrite it (input-boundary copy via a non-dispatch path). Feed "
            "this leaf through a GPU op, or exclude it from the input set.");
      }
    }
  }

  // 1. Pack every setBytes payload into one shared buffer at 256-B-aligned
  //    slots and remember each slot's offset (indirect commands have no
  //    setBytes — these become kernel-buffer binds).
  size_t total_args = 0;
  std::vector<std::vector<size_t>> bytes_offsets(dispatches.size());
  for (size_t i = 0; i < dispatches.size(); ++i) {
    bytes_offsets[i].reserve(dispatches[i].bytes.size());
    for (auto& b : dispatches[i].bytes) {
      bytes_offsets[i].push_back(total_args);
      total_args += align_up(b.data.size(), kArgAlign);
    }
  }
  size_t args_size = std::max<size_t>(total_args, kArgAlign);
  impl->packed_args = NS::TransferPtr(
      mtl_device->newBuffer(args_size, MTL::ResourceStorageModeShared));
  if (!impl->packed_args) {
    fail("[CaptureReplay::capture] Failed to allocate packed args buffer.");
  }
  {
    auto* base = static_cast<uint8_t*>(impl->packed_args->contents());
    std::memset(base, 0, args_size);
    for (size_t i = 0; i < dispatches.size(); ++i) {
      for (size_t j = 0; j < dispatches[i].bytes.size(); ++j) {
        auto& b = dispatches[i].bytes[j];
        if (!b.data.empty()) {
          std::memcpy(base + bytes_offsets[i][j], b.data.data(), b.data.size());
        }
      }
    }
  }

  // 2. Re-create each distinct captured pipeline with
  //    supportIndirectCommandBuffers=YES from its retained MTL::Function.
  std::unordered_map<MTL::ComputePipelineState*, MTL::ComputePipelineState*>
      icb_of;
  uint32_t max_buffer_index = 0;
  uint32_t max_tgmem_index = 0;
  bool has_tgmem = false;
  for (auto& disp : dispatches) {
    for (auto& bb : disp.buffers) {
      max_buffer_index = std::max(max_buffer_index, bb.index);
    }
    for (auto& by : disp.bytes) {
      max_buffer_index = std::max(max_buffer_index, by.index);
    }
    for (auto& tg : disp.tgmem) {
      max_tgmem_index = std::max(max_tgmem_index, tg.index);
      has_tgmem = true;
    }
    if (icb_of.count(disp.pipeline)) {
      continue;
    }
    MTL::Function* fn = capture::function_for_pipeline(disp.pipeline);
    if (fn == nullptr) {
      fail(
          "[CaptureReplay::capture] A captured pipeline has no retained "
          "MTL::Function (kernel built before MLX_CAPTURE_REPLAY was set, or "
          "built with linked_functions). Cannot rebuild it for the ICB.");
    }
    auto desc =
        NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    desc->setComputeFunction(fn);
    desc->setSupportIndirectCommandBuffers(true);
    // Blocker 3: rebuild with the SAME linked (private) functions the kernel was
    // originally built from, or newComputePipelineState fails to resolve the
    // private symbols. Empty for plain kernels (alpha toy) → no behaviour change.
    auto linked = capture::linked_functions_for_pipeline(disp.pipeline);
    NS::SharedPtr<MTL::LinkedFunctions> lfuncs;
    if (!linked.empty()) {
      lfuncs = NS::TransferPtr(MTL::LinkedFunctions::linkedFunctions());
      NS::Array* funcs_arr = NS::Array::array(
          reinterpret_cast<const NS::Object* const*>(linked.data()),
          linked.size());
      lfuncs->setPrivateFunctions(funcs_arr);
      desc->setLinkedFunctions(lfuncs.get());
    }
    NS::Error* error = nullptr;
    auto icb_pipe = NS::TransferPtr(mtl_device->newComputePipelineState(
        desc.get(), MTL::PipelineOptionNone, nullptr, &error));
    if (!icb_pipe) {
      std::string emsg = error
          ? std::string(error->localizedDescription()->utf8String())
          : std::string("unknown error");
      fail(
          "[CaptureReplay::capture] Failed to build ICB-capable pipeline: " +
          emsg);
    }
    icb_of[disp.pipeline] = icb_pipe.get();
    impl->icb_pipelines.push_back(icb_pipe);
  }

  // 3. Build the ICB.
  auto icb_desc =
      NS::TransferPtr(MTL::IndirectCommandBufferDescriptor::alloc()->init());
  icb_desc->setCommandTypes(
      MTL::IndirectCommandTypeConcurrentDispatch |
      MTL::IndirectCommandTypeConcurrentDispatchThreads);
  icb_desc->setInheritBuffers(false);
  icb_desc->setInheritPipelineState(false);
  icb_desc->setMaxKernelBufferBindCount(max_buffer_index + 1);
  if (has_tgmem) {
    icb_desc->setMaxKernelThreadgroupMemoryBindCount(max_tgmem_index + 1);
  }
  impl->icb = NS::TransferPtr(mtl_device->newIndirectCommandBuffer(
      icb_desc.get(), dispatches.size(), MTL::ResourceStorageModeShared));
  if (!impl->icb) {
    fail("[CaptureReplay::capture] Failed to allocate indirect command buffer.");
  }

  // Deferred barrier pruning (replaces the alpha's per-command barrier). The
  // alpha put setBarrier() on every command, fully serializing the stream on the
  // GPU. We set a barrier ONLY where a true hazard exists: a command that
  // touches a buffer WRITTEN by any command since the previous barrier. A
  // barrier makes all prior commands complete before it, so it clears the
  // accumulated write set; independent commands between barriers run
  // concurrently. A pruning miss surfaces as a bitwise mismatch in the bench.
  //
  // Optional renaming (fix a, env MLX_CAPTURE_REPLAY_RENAME): a WAW hazard on a
  // physical buffer that an independent later command FULLY overwrites (donation
  // reuse across independent ops) is a false chain — the buffers are logically
  // distinct. We break it by giving that write its own duplicate buffer and
  // rebinding subsequent reads of that version, converting the false chain into
  // real independence. Conservative triggers only (never an input/output buffer,
  // only a bound-ONCE write that reuses a previously-written buffer), so in-place
  // RMW (bound twice) keeps its true ordering. CAVEAT: a bound-once write that
  // covers only PART of a reused buffer (e.g. per-slice concat copies into one
  // output) would be wrongly renamed — hence rename is EXPERIMENTAL and gated:
  // the bench's bitwise gates verify every rename, and it is default OFF (the
  // pruned-only path is already GPU-validated). Diagnostics below quantify how
  // many barriers renaming WOULD remove even when it is off.
  const bool rename_enabled = [] {
    const char* v = std::getenv("MLX_CAPTURE_REPLAY_RENAME");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();

  // Output/input buffer sets — never renamed (keeps copy-out + replay writes
  // targeting the buffers the caller reads / rewrites).
  std::unordered_set<const MTL::Buffer*> output_bufs;
  for (auto& o : outputs) {
    output_bufs.insert(static_cast<const MTL::Buffer*>(o.buffer().ptr()));
  }

  // Effective (post-rename) binds and write set per dispatch. Identity when
  // rename is off. cur maps an original buffer to its current live version.
  std::vector<std::vector<capture::BufferBind>> eff_binds(dispatches.size());
  std::vector<std::vector<const MTL::Buffer*>> eff_writes(dispatches.size());
  std::unordered_map<const MTL::Buffer*, const MTL::Buffer*> cur;
  std::unordered_set<const MTL::Buffer*> written_before;
  std::unordered_map<const MTL::Buffer*, size_t> write_freq;
  size_t renamable = 0;
  size_t renamed = 0;

  auto current_of = [&cur](const MTL::Buffer* b) -> const MTL::Buffer* {
    auto it = cur.find(b);
    return it == cur.end() ? b : it->second;
  };

  for (size_t i = 0; i < dispatches.size(); ++i) {
    auto& disp = dispatches[i];
    // Count bind occurrences to detect bound-once (pure) writes vs RMW.
    std::unordered_map<const MTL::Buffer*, int> occ;
    for (auto& bb : disp.buffers) {
      occ[bb.buf]++;
    }
    // Decide renames for this dispatch's writes.
    std::unordered_map<const MTL::Buffer*, const MTL::Buffer*> renamed_here;
    for (auto* wb : disp.write_bufs) {
      write_freq[wb]++;
      const bool reuse = written_before.count(wb) != 0;
      const bool pure_write = occ[wb] == 1; // bound once -> not RMW
      const bool safe = pure_write && output_bufs.find(wb) == output_bufs.end();
      if (reuse && safe) {
        ++renamable;
        if (rename_enabled) {
          auto dup = NS::TransferPtr(mtl_device->newBuffer(
              wb->length(), MTL::ResourceStorageModeShared));
          if (dup) {
            renamed_here[wb] = dup.get();
            cur[wb] = dup.get();
            impl->rename_dupes.push_back(dup);
            ++renamed;
          }
        }
      }
    }
    // Build effective binds: a written buffer takes its renamed version (if
    // any); every other bind takes the current live version.
    eff_binds[i].reserve(disp.buffers.size());
    for (auto& bb : disp.buffers) {
      const MTL::Buffer* eff = bb.buf;
      auto rit = renamed_here.find(bb.buf);
      if (rit != renamed_here.end()) {
        eff = rit->second;
      } else {
        eff = current_of(bb.buf);
      }
      eff_binds[i].push_back(capture::BufferBind{eff, bb.offset, bb.index});
    }
    eff_writes[i].reserve(disp.write_bufs.size());
    for (auto* wb : disp.write_bufs) {
      eff_writes[i].push_back(current_of(wb));
    }
    for (auto* wb : disp.write_bufs) {
      written_before.insert(wb);
    }
  }
  for (auto& kv : write_freq) {
    impl->max_buffer_write_count =
        std::max(impl->max_buffer_write_count, kv.second);
  }

  std::unordered_set<const MTL::Buffer*> write_window;
  std::unordered_map<const MTL::Buffer*, size_t> barrier_cause;
  size_t barrier_count = 0;
  size_t raw_barriers = 0;
  size_t waw_barriers = 0;
  size_t current_run = 0;
  size_t largest_run = 0;
  for (size_t i = 0; i < dispatches.size(); ++i) {
    auto& disp = dispatches[i];
    auto* cmd = impl->icb->indirectComputeCommand(i);
    cmd->setComputePipelineState(icb_of[disp.pipeline]);
    for (auto& bb : eff_binds[i]) {
      cmd->setKernelBuffer(bb.buf, bb.offset, bb.index);
    }
    for (size_t j = 0; j < disp.bytes.size(); ++j) {
      cmd->setKernelBuffer(
          impl->packed_args.get(), bytes_offsets[i][j], disp.bytes[j].index);
    }
    for (auto& tg : disp.tgmem) {
      cmd->setThreadgroupMemoryLength(tg.length, tg.index);
    }
    if (disp.uses_threads) {
      cmd->concurrentDispatchThreads(disp.grid, disp.group);
    } else {
      cmd->concurrentDispatchThreadgroups(disp.grid, disp.group);
    }

    bool read_hazard = false;
    bool write_hazard = false;
    const MTL::Buffer* cause = nullptr;
    if (!write_window.empty()) {
      for (auto& bb : eff_binds[i]) { // read set = every bound buffer
        if (write_window.count(bb.buf)) {
          read_hazard = true;
          cause = bb.buf;
          break;
        }
      }
      if (!read_hazard) {
        for (auto* wb : eff_writes[i]) { // WAW against the write window
          if (write_window.count(wb)) {
            write_hazard = true;
            cause = wb;
            break;
          }
        }
      }
    }
    if (read_hazard || write_hazard) {
      cmd->setBarrier();
      ++barrier_count;
      raw_barriers += read_hazard ? 1 : 0;
      waw_barriers += write_hazard ? 1 : 0;
      if (cause) {
        barrier_cause[cause]++;
      }
      write_window.clear();
      largest_run = std::max(largest_run, current_run);
      current_run = 0;
    }
    ++current_run;
    for (auto* wb : eff_writes[i]) {
      write_window.insert(wb);
    }
  }
  largest_run = std::max(largest_run, current_run);
  for (auto& kv : barrier_cause) {
    impl->hottest_barrier_buffer_barriers =
        std::max(impl->hottest_barrier_buffer_barriers, kv.second);
  }
  impl->num_barriers = barrier_count;
  impl->largest_barrier_free_run = largest_run;
  impl->num_raw_barriers = raw_barriers;
  impl->num_waw_barriers = waw_barriers;
  impl->renamable_writes = renamable;
  impl->renamed_writes = renamed;

  // 4. Dedicated command queue for replay submission.
  impl->queue = NS::TransferPtr(mtl_device->newCommandQueue());
  if (!impl->queue) {
    fail("[CaptureReplay::capture] Failed to create replay command queue.");
  }

  // 4b. Async-replay amortization: a persistent MTLResidencySet on the replay
  // queue makes the arena/heaps/dupes/packed_args resident ONCE, so
  // replay_submit skips the per-call useResources/useHeap host cost (the S3
  // lever). This declares residency over already-allocated buffers — it does not
  // allocate, so it does not move the allocator's memory knob. macOS 15+/Metal3
  // only; on older systems use_residency stays false and replay_submit does the
  // per-call residency (identical to the synchronous path). Cache the arena as a
  // Resource* array for that fallback either way.
  impl->arena_resources.reserve(impl->arena.size());
  for (auto* b : impl->arena) {
    impl->arena_resources.push_back(static_cast<const MTL::Resource*>(b));
  }
  if (__builtin_available(macOS 15, iOS 18, *)) {
    if (mtl_device->supportsFamily(MTL::GPUFamilyMetal3)) {
      auto rdesc =
          NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
      NS::Error* rerr = nullptr;
      auto rset = NS::TransferPtr(mtl_device->newResidencySet(rdesc.get(), &rerr));
      if (rset) {
        for (auto* b : impl->arena) {
          rset->addAllocation(b);
        }
        for (auto* h : impl->heaps) {
          rset->addAllocation(h);
        }
        for (auto& dup : impl->rename_dupes) {
          rset->addAllocation(dup.get());
        }
        rset->addAllocation(impl->packed_args.get());
        rset->commit();
        rset->requestResidency();
        impl->queue->addResidencySet(rset.get());
        impl->residency = rset;
        impl->use_residency = true;
      }
    }
  }

  // 5. Record the captured leaves / results (they hold the pinned buffers).
  impl->inputs = inputs;
  impl->outputs = outputs;
  impl->input_bufs.reserve(inputs.size());
  for (auto& in : inputs) {
    impl->input_bufs.push_back(
        static_cast<const MTL::Buffer*>(in.buffer().ptr()));
  }

  return std::shared_ptr<CaptureReplay>(new CaptureReplay(std::move(impl)));
}

uint64_t CaptureReplay::replay_submit(const std::vector<array>& new_inputs) {
  auto& impl = *impl_;
  if (new_inputs.size() != impl.inputs.size()) {
    throw std::invalid_argument(
        "[CaptureReplay::replay_submit] Expected " +
        std::to_string(impl.inputs.size()) + " inputs, got " +
        std::to_string(new_inputs.size()) + ".");
  }
  std::lock_guard<std::mutex> lk(impl.submit_mtx);
  auto pool = metal::new_scoped_memory_pool();

  // (a) Address-stability assert + (b) write new values into pinned inputs.
  // NOTE: at pipeline depth > 1 these writes can race a prior in-flight replay's
  // GPU reads of the same pinned input buffers; the bench uses same-value inputs
  // (byte-identical writes are race-free) for pipelined timing and proves
  // fresh-value correctness unpipelined. Documented in the bench.
  for (size_t i = 0; i < new_inputs.size(); ++i) {
    auto* cur = static_cast<const MTL::Buffer*>(impl.inputs[i].buffer().ptr());
    if (cur != impl.input_bufs[i]) {
      throw std::runtime_error(
          "[CaptureReplay::replay_submit] Captured input buffer moved — the "
          "arena is no longer address-stable (allocator reuse).");
    }
    array ni = new_inputs[i];
    ni.eval();
    if (ni.nbytes() != impl.inputs[i].nbytes()) {
      throw std::invalid_argument(
          "[CaptureReplay::replay_submit] Input " + std::to_string(i) +
          " size mismatch vs capture.");
    }
    std::memcpy(
        impl.inputs[i].data<char>(), ni.data<const char>(), ni.nbytes());
  }

  // (c) Encode ONE command buffer and commit WITHOUT waiting. When the
  // persistent residency set is active the per-call residency declaration is
  // skipped (the amortization); otherwise fall back to per-call residency.
  auto* cb = impl.queue->commandBuffer();
  auto* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  if (!impl.use_residency) {
    if (!impl.arena_resources.empty()) {
      enc->useResources(
          impl.arena_resources.data(),
          impl.arena_resources.size(),
          MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    for (auto* h : impl.heaps) {
      enc->useHeap(h);
    }
    for (auto& dup : impl.rename_dupes) {
      enc->useResource(
          dup.get(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
    enc->useResource(impl.packed_args.get(), MTL::ResourceUsageRead);
  }
  enc->executeCommandsInBuffer(impl.icb.get(), NS::Range(0, impl.num_commands));
  enc->endEncoding();
  cb->commit();

  uint64_t ticket = impl.next_ticket++;
  impl.inflight.emplace(ticket, NS::RetainPtr(cb));
  return ticket;
}

void CaptureReplay::replay_wait(uint64_t ticket) {
  auto& impl = *impl_;
  auto pool = metal::new_scoped_memory_pool();
  NS::SharedPtr<MTL::CommandBuffer> cb;
  {
    std::lock_guard<std::mutex> lk(impl.submit_mtx);
    auto it = impl.inflight.find(ticket);
    if (it == impl.inflight.end()) {
      throw std::invalid_argument(
          "[CaptureReplay::replay_wait] Unknown or already-waited ticket " +
          std::to_string(ticket) + ".");
    }
    cb = it->second;
    impl.inflight.erase(it);
  }
  cb->waitUntilCompleted();
  if (cb->status() == MTL::CommandBufferStatusError) {
    std::string emsg = cb->error()
        ? std::string(cb->error()->localizedDescription()->utf8String())
        : std::string("unknown error");
    throw std::runtime_error(
        "[CaptureReplay::replay_wait] Command buffer failed: " + emsg);
  }
}

std::vector<array> CaptureReplay::read_outputs() {
  auto& impl = *impl_;
  // Copy each pinned output into a fresh host-backed array. Call after the
  // producing replay has completed (replay_wait); the pinned buffer is
  // overwritten by the next replay. Excluded from the async timed path.
  std::vector<array> results;
  results.reserve(impl.outputs.size());
  for (auto& out : impl.outputs) {
    size_t nbytes = out.nbytes();
    void* mem = std::malloc(std::max<size_t>(nbytes, 1));
    std::memcpy(mem, out.data<const char>(), nbytes);
    results.emplace_back(
        mem, out.shape(), out.dtype(), [](void* p) { std::free(p); });
  }
  return results;
}

bool CaptureReplay::residency_set_active() const {
  return impl_->use_residency;
}

std::vector<array> CaptureReplay::replay(const std::vector<array>& new_inputs) {
  // Compat synchronous path: submit + wait + copy-out (unchanged behaviour for
  // the toy and the existing bench phases).
  uint64_t ticket = replay_submit(new_inputs);
  replay_wait(ticket);
  return read_outputs();
}

} // namespace mlx::core::metal
