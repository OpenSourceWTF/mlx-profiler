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
//   * every command gets setBarrier() — correctness first; the concurrent
//     encoder + per-command barrier is a strict superset of MLX's dependency
//     edges (dependency-pruning is future work).
// Every MTLBuffer the stream touches is pinned in the allocator so its address
// is stable; replay() writes new input values into the pinned input buffers and
// issues a single executeCommandsInBuffer.
//
// Alpha invariants NOT yet handled (these WILL break a graph):
//   * Allocator reuse across evals: pinning holds captured buffers, but any
//     graph whose per-cycle buffer set is not identical to the captured one
//     (dynamic shapes, KV-cache growth that reallocates, control flow) is out
//     of scope — the two compiled entries in the A3B lane are shape-stable, the
//     toy alpha graph is fixed.
//   * setBytes values are baked as CONSTANTS. Any per-cycle-varying value that
//     MLX passes via setBytes (rather than a tensor buffer) would be frozen at
//     capture. The R1/compiled A3B stack already makes cycle-varying values
//     tensor-resident; the toy graph has none.
//   * Kernels created with linked_functions (rare; not in the toy graph) are
//     not registered for ICB pipeline re-creation → capture() throws.
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
std::mutex g_fn_mtx;
std::unordered_map<MTL::ComputePipelineState*, NS::SharedPtr<MTL::Function>>
    g_functions;

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
    MTL::Function* fn) {
  if (!detail::g_enabled || pipeline == nullptr || fn == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lk(g_fn_mtx);
  g_functions.emplace(pipeline, NS::RetainPtr(fn));
}

MTL::Function* function_for_pipeline(MTL::ComputePipelineState* pipeline) {
  std::lock_guard<std::mutex> lk(g_fn_mtx);
  auto it = g_functions.find(pipeline);
  return it == g_functions.end() ? nullptr : it->second.get();
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

  // Distinct pinned buffers (arena) — used for useResources and, at teardown,
  // for the deferred allocator free.
  std::vector<MTL::Buffer*> arena;

  // Captured graph leaves / results. inputs are rewritten on replay; outputs
  // are copied out. input_bufs mirrors the input buffers for the
  // address-stability assert.
  std::vector<array> inputs;
  std::vector<array> outputs;
  std::vector<const MTL::Buffer*> input_bufs;

  ~Impl() {
    // Drop MTL objects first.
    icb.reset();
    packed_args.reset();
    icb_pipelines.clear();
    queue.reset();
    // Destroy the captured arrays BEFORE unpinning: their Data destructors call
    // allocator.free() on the pinned buffers, which defers (no-op) while still
    // pinned. Only after the arrays are gone do we unpin — that performs the
    // single real free (recycle to cache) with no other owner left.
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

  for (size_t i = 0; i < dispatches.size(); ++i) {
    auto& disp = dispatches[i];
    auto* cmd = impl->icb->indirectComputeCommand(i);
    cmd->setComputePipelineState(icb_of[disp.pipeline]);
    for (auto& bb : disp.buffers) {
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
    // Alpha: one barrier per command — correctness over throughput.
    cmd->setBarrier();
  }

  // 4. Dedicated command queue for replay submission.
  impl->queue = NS::TransferPtr(mtl_device->newCommandQueue());
  if (!impl->queue) {
    fail("[CaptureReplay::capture] Failed to create replay command queue.");
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

std::vector<array> CaptureReplay::replay(const std::vector<array>& new_inputs) {
  auto& impl = *impl_;
  if (new_inputs.size() != impl.inputs.size()) {
    throw std::invalid_argument(
        "[CaptureReplay::replay] Expected " +
        std::to_string(impl.inputs.size()) + " inputs, got " +
        std::to_string(new_inputs.size()) + ".");
  }

  auto pool = metal::new_scoped_memory_pool();

  // (a) Address-stability assert + (b) write new values into pinned inputs.
  for (size_t i = 0; i < new_inputs.size(); ++i) {
    // vLLM-style assert: the pinned input buffer must not have moved.
    auto* cur = static_cast<const MTL::Buffer*>(impl.inputs[i].buffer().ptr());
    if (cur != impl.input_bufs[i]) {
      throw std::runtime_error(
          "[CaptureReplay::replay] Captured input buffer moved — the arena is "
          "no longer address-stable (allocator reuse). Alpha cannot recover.");
    }
    array ni = new_inputs[i];
    ni.eval();
    if (ni.nbytes() != impl.inputs[i].nbytes()) {
      throw std::invalid_argument(
          "[CaptureReplay::replay] Input " + std::to_string(i) +
          " size mismatch vs capture.");
    }
    std::memcpy(
        impl.inputs[i].data<char>(), ni.data<const char>(), ni.nbytes());
  }

  // (c) Submit ONE command buffer: make the arena resident, execute the ICB.
  auto* cb = impl.queue->commandBuffer();
  auto* enc = cb->computeCommandEncoder(MTL::DispatchTypeConcurrent);
  if (!impl.arena.empty()) {
    enc->useResources(
        reinterpret_cast<const MTL::Resource* const*>(impl.arena.data()),
        impl.arena.size(),
        MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  enc->useResource(
      impl.packed_args.get(), MTL::ResourceUsageRead);
  enc->executeCommandsInBuffer(impl.icb.get(), NS::Range(0, impl.num_commands));
  enc->endEncoding();
  cb->commit();
  cb->waitUntilCompleted();
  if (cb->status() == MTL::CommandBufferStatusError) {
    std::string emsg = cb->error()
        ? std::string(cb->error()->localizedDescription()->utf8String())
        : std::string("unknown error");
    throw std::runtime_error(
        "[CaptureReplay::replay] Command buffer failed: " + emsg);
  }

  // (d) Copy each pinned output into a fresh host-backed array so the caller
  //     holds a stable snapshot (the pinned buffer is overwritten next replay).
  std::vector<array> results;
  results.reserve(impl.outputs.size());
  for (auto& out : impl.outputs) {
    size_t nbytes = out.nbytes();
    void* mem = std::malloc(std::max<size_t>(nbytes, 1));
    std::memcpy(mem, out.data<const char>(), nbytes);
    results.emplace_back(
        mem,
        out.shape(),
        out.dtype(),
        [](void* p) { std::free(p); });
  }
  return results;
}

} // namespace mlx::core::metal
