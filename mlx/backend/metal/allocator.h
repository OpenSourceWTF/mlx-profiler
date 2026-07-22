// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <map>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

using allocator::Buffer;

class MetalAllocator : public allocator::Allocator {
  /** Allocator for Metal GPUs. */
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;
  virtual Buffer make_buffer(void* ptr, size_t size) override;
  virtual void release(Buffer buffer) override;

  size_t get_active_memory() {
    return active_memory_;
  };
  size_t get_peak_memory() {
    return peak_memory_;
  };
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  };
  size_t get_cache_memory() {
    return buffer_cache_.cache_size();
  };
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit);
  size_t get_memory_limit();
  size_t set_wired_limit(size_t limit);
  void clear_cache();

  // --- S2b capture-replay pinning (alpha) ---------------------------------
  // A pinned buffer is excluded from the reuse cache and never released while
  // pinned: its address is stable for the lifetime of a CaptureReplay handle.
  // pin() marks a buffer that is still owned by a live array; a later free()
  // for that buffer is deferred (recorded, skipped) until unpin() performs the
  // real free. This forbids allocator reuse/donation of any buffer the captured
  // ICB references. See capture_replay.cpp.
  //
  // S2b-beta: unpin() performs the real free ONLY for buffers whose free() was
  // actually deferred while pinned. The real A3B graph pins many buffers that
  // stay owned by live arrays (model weights bound by matmul/qmm dispatches) and
  // are never freed during the handle's life; unpinning those must be a no-op,
  // or the still-owned buffer would be recycled/released out from under its
  // owner. The toy graph only pins buffers it exclusively owns, so its behaviour
  // is unchanged.
  void pin(MTL::Buffer* buf);
  void unpin(MTL::Buffer* buf);

 private:
  MTL::Device* device_;

  // The size of allocations which go on the heap until it is full. This size
  // is chosen because it is the actual minimum size of a buffer allocated from
  // the heap, a heap can have at most heap.size() / 256 buffers.
  static constexpr int small_size_ = 256;
  static constexpr int heap_size_ = 1 << 20;

  MetalAllocator(Device& d);
  ~MetalAllocator();

  friend MetalAllocator& allocator();

  NS::SharedPtr<MTL::Heap> heap_;
  ResidencySet& residency_set_;

  // Caching allocator
  BufferCache<MTL::Buffer> buffer_cache_;

  // Allocation stats
  size_t block_limit_;
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t max_pool_size_;
  size_t wired_limit_{0};
  size_t num_resources_{0};
  size_t resource_limit_{0};

  // Buffers pinned by an active capture (guarded by mutex_).
  std::unordered_set<MTL::Buffer*> pinned_;
  // Pinned buffers whose free() arrived while pinned and was deferred. unpin()
  // frees exactly these (buffers the capture is the last owner of); a pinned
  // buffer absent here is still owned elsewhere and unpin() leaves it be.
  std::unordered_set<MTL::Buffer*> deferred_free_;

  std::mutex mutex_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
