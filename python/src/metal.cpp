// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/transforms.h"
#include "python/src/small_vector.h"
#include "python/src/trees.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &mx::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("mx.metal.get_active_memory", "mx.get_active_memory");
    return mx::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("mx.metal.get_peak_memory", "mx.get_peak_memory");
    return mx::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("mx.metal.get_cache_memory", "mx.get_cache_memory");
    return mx::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_memory_limit", "mx.set_memory_limit");
        return mx::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_cache_limit", "mx.set_cache_limit");
        return mx::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_wired_limit", "mx.set_wired_limit");
        return mx::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
  });
  metal.def(
      "start_capture",
      &mx::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &mx::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def("device_info", []() {
    DEPRECATE("mx.metal.device_info", "mx.device_info");
    return mx::device_info(mx::Device(mx::Device::gpu, 0));
  });

  // --- S2b capture-replay (alpha) ------------------------------------------
  // A validated device-side feedback plan (S3): P_out -> P_in blits appended to
  // the replay command buffer. Opaque handle; build it with
  // CaptureReplay.make_feedback_plan and pass it back to a submit variant.
  nb::class_<mx::metal::CaptureReplay::FeedbackPlan>(
      metal,
      "CaptureReplayFeedbackPlan",
      R"pbdoc(
      EXPERIMENTAL (S3). A validated list of (output-leaf -> input-leaf) blits
      appended to a replay command buffer after the ICB, advancing decode state
      on-device with zero host bytes. Create it with
      :meth:`CaptureReplay.make_feedback_plan`.
      )pbdoc")
      .def_prop_ro(
          "num_blits",
          [](const mx::metal::CaptureReplay::FeedbackPlan& self) {
            return self.blits.size();
          },
          "Number of feedback blits in the plan.");

  nb::class_<mx::metal::CaptureReplay>(
      metal,
      "CaptureReplay",
      R"pbdoc(
      EXPERIMENTAL (S2b alpha). A replayable handle over the compute dispatch
      stream of one eval of a compiled graph, baked into an
      MTLIndirectCommandBuffer. Create it with :func:`capture_compiled`.
      )pbdoc")
      .def(
          "replay",
          [](mx::metal::CaptureReplay& self, std::vector<mx::array> inputs) {
            nb::gil_scoped_release nogil;
            return self.replay(inputs);
          },
          "inputs"_a,
          R"pbdoc(
          Write ``inputs`` into the pinned input buffers (address-stability
          asserted) and submit one command buffer that executes the ICB.
          Returns a fresh copy of each pinned output array (submit + wait + read).
          )pbdoc")
      .def(
          "make_feedback_plan",
          [](mx::metal::CaptureReplay& self,
             std::vector<std::pair<size_t, size_t>> pairs) {
            nb::gil_scoped_release nogil;
            return self.make_feedback_plan(pairs);
          },
          "pairs"_a,
          R"pbdoc(
          Validate and build a :class:`CaptureReplayFeedbackPlan` from a list of
          ``(output_leaf_index, input_leaf_index)`` pairs. Each pair must be in
          range and the two leaves byte-size-equal (raises otherwise). Pass the
          returned plan to :meth:`replay_submit` / :meth:`replay_submit_partial`
          to blit each state output straight into its state input on-device.
          )pbdoc")
      .def(
          "replay_submit",
          [](mx::metal::CaptureReplay& self,
             std::vector<mx::array> inputs,
             std::shared_ptr<mx::metal::CaptureReplay::FeedbackPlan> plan) {
            nb::gil_scoped_release nogil;
            return self.replay_submit(inputs, plan);
          },
          "inputs"_a,
          "feedback_plan"_a = nb::none(),
          R"pbdoc(
          Write ``inputs`` into the pinned buffers and commit ONE command buffer
          WITHOUT waiting. Returns an integer ticket for :func:`replay_wait`.
          At pipeline depth > 1 successive submits share the pinned input
          buffers; use same-value inputs or external double-buffering.
          A non-null ``feedback_plan`` appends its P_out->P_in blits to the same
          command buffer (fenced after the ICB); feedback submits must be serial.
          )pbdoc")
      .def(
          "replay_submit_partial",
          [](mx::metal::CaptureReplay& self,
             std::vector<size_t> indices,
             std::vector<mx::array> arrays,
             std::shared_ptr<mx::metal::CaptureReplay::FeedbackPlan> plan) {
            nb::gil_scoped_release nogil;
            return self.replay_submit_partial(indices, arrays, plan);
          },
          "indices"_a,
          "arrays"_a,
          "feedback_plan"_a = nb::none(),
          R"pbdoc(
          Rewrite ONLY the input leaves in ``indices`` (paired with ``arrays``)
          into their pinned buffers; every other pinned input is left untouched
          (it persists the previous submit / feedback-blit contents). Same ticket
          / wait semantics as :meth:`replay_submit`. Address stability is asserted
          on the touched leaves (and, with a ``feedback_plan``, on its blit src/
          dst). Returns a ticket for :func:`replay_wait`.
          )pbdoc")
      .def(
          "write_input_range",
          [](mx::metal::CaptureReplay& self,
             size_t index,
             size_t byte_offset,
             mx::array src) {
            nb::gil_scoped_release nogil;
            self.write_input_range(index, byte_offset, src);
          },
          "index"_a,
          "byte_offset"_a,
          "src"_a,
          R"pbdoc(
          Sub-leaf range write: memcpy ``src`` (contiguous) into pinned input leaf
          ``index`` at ``byte_offset`` within that leaf's data, leaving every other
          byte — and every other leaf — untouched. Address stability is asserted;
          ``byte_offset + src.nbytes`` must fit the leaf. Does NOT commit — pair
          range writes with a following :meth:`replay_submit_partial` (small
          cycle-varying leaves + commit). Used by the S3 draft delta-rewrite to
          replace the reserved-window K/V memcpy with just the 1-2 appended rows.
          Serial-use only, like :meth:`replay_submit_partial`.
          )pbdoc")
      .def(
          "replay_wait",
          [](mx::metal::CaptureReplay& self, uint64_t ticket) {
            nb::gil_scoped_release nogil;
            self.replay_wait(ticket);
          },
          "ticket"_a,
          "Block until the command buffer for ``ticket`` completes.")
      .def(
          "read_outputs",
          [](mx::metal::CaptureReplay& self) {
            nb::gil_scoped_release nogil;
            return self.read_outputs();
          },
          R"pbdoc(
          Copy the current pinned output buffers into fresh host arrays. Call
          after the producing replay has been waited; excluded from timed paths.
          )pbdoc")
      .def(
          "read_output",
          [](mx::metal::CaptureReplay& self, size_t index) {
            nb::gil_scoped_release nogil;
            return self.read_output(index);
          },
          "index"_a,
          R"pbdoc(
          Copy ONE pinned output leaf into a fresh host array (the selective-read
          counterpart of :meth:`read_outputs`). Call after the producing replay
          has been waited.
          )pbdoc")
      .def(
          "read_input",
          [](mx::metal::CaptureReplay& self, size_t index) {
            nb::gil_scoped_release nogil;
            return self.read_input(index);
          },
          "index"_a,
          R"pbdoc(
          Copy ONE pinned INPUT leaf into a fresh host array, addressed through
          the array's data pointer (respecting its offset), so it returns the
          leaf's current logical contents — including bytes a feedback blit wrote
          into it. Diagnostic counterpart of :meth:`read_output`: read back the
          fed-back state inputs to isolate blit correctness from execution /
          reference. Call after the producing replay has been waited.
          )pbdoc")
      .def(
          "input_buffer_info",
          [](mx::metal::CaptureReplay& self, size_t index) {
            auto info = self.input_buffer_info(index);
            nb::dict d;
            d["buffer_id"] = info.buffer_id;
            d["offset"] = info.offset;
            d["nbytes"] = info.nbytes;
            d["buffer_length"] = info.buffer_length;
            d["heap"] = info.heap;
            return d;
          },
          "index"_a,
          R"pbdoc(
          Debug view of an INPUT leaf's backing buffer as a dict: ``buffer_id``
          (opaque MTL::Buffer identity), ``offset`` (the array's byte offset
          within the buffer — nonzero for a view / suballocated leaf), ``nbytes``
          (logical leaf size), ``buffer_length`` (allocation length), and
          ``heap`` (suballocated from the shared < 256 B heap). Reveals which
          state leaves sit at a nonzero offset — the case a naive offset-0 blit
          would mis-copy.
          )pbdoc")
      .def(
          "output_buffer_info",
          [](mx::metal::CaptureReplay& self, size_t index) {
            auto info = self.output_buffer_info(index);
            nb::dict d;
            d["buffer_id"] = info.buffer_id;
            d["offset"] = info.offset;
            d["nbytes"] = info.nbytes;
            d["buffer_length"] = info.buffer_length;
            d["heap"] = info.heap;
            return d;
          },
          "index"_a,
          R"pbdoc(
          Debug view of an OUTPUT leaf's backing buffer (same dict shape as
          :meth:`input_buffer_info`). Reveals which final_state outputs sit at a
          nonzero offset — the blit source side of the same mis-copy case.
          )pbdoc")
      .def(
          "output_arrays",
          [](mx::metal::CaptureReplay& self, std::vector<size_t> indices) {
            nb::gil_scoped_release nogil;
            return self.output_arrays(indices);
          },
          "indices"_a,
          R"pbdoc(
          Zero-copy mx.array views of the pinned output buffers at ``indices``.
          ALIASING CONTRACT: the views alias the pinned buffers in place and
          reflect the most recently completed replay; they are only valid until
          the next replay submit overwrites those buffers. Consume or copy them
          before resubmitting. No host bytes are moved.
          )pbdoc")
      .def_prop_ro(
          "residency_set_active",
          &mx::metal::CaptureReplay::residency_set_active,
          "Whether the amortized persistent-residency path is active (mac15+).")
      .def_prop_ro(
          "num_commands",
          &mx::metal::CaptureReplay::num_commands,
          "Number of compute commands baked into the ICB.")
      .def_prop_ro(
          "num_barriers",
          &mx::metal::CaptureReplay::num_barriers,
          "Commands carrying a setBarrier() after deferred barrier pruning "
          "(== num_commands for a fully serial / fully dependent stream).")
      .def_prop_ro(
          "largest_barrier_free_run",
          &mx::metal::CaptureReplay::largest_barrier_free_run,
          "Longest run of consecutive barrier-free (concurrent) commands.")
      .def_prop_ro(
          "num_raw_barriers",
          &mx::metal::CaptureReplay::num_raw_barriers,
          "Barriers triggered by a read-after-write hazard.")
      .def_prop_ro(
          "num_waw_barriers",
          &mx::metal::CaptureReplay::num_waw_barriers,
          "Barriers triggered by a write-after-write hazard.")
      .def_prop_ro(
          "max_buffer_write_count",
          &mx::metal::CaptureReplay::max_buffer_write_count,
          "Writes to the single hottest buffer (a shared scratch shows here).")
      .def_prop_ro(
          "hottest_barrier_buffer_barriers",
          &mx::metal::CaptureReplay::hottest_barrier_buffer_barriers,
          "Barriers caused by the single worst-offending buffer.")
      .def_prop_ro(
          "renamable_writes",
          &mx::metal::CaptureReplay::renamable_writes,
          "Full-def reuse writes that renaming could break into independence.")
      .def_prop_ro(
          "renamed_writes",
          &mx::metal::CaptureReplay::renamed_writes,
          "Writes actually renamed (0 unless MLX_CAPTURE_REPLAY_RENAME set).")
      .def_prop_ro(
          "inputs",
          &mx::metal::CaptureReplay::inputs,
          "The captured input arrays (pinned buffers).")
      .def_prop_ro(
          "outputs",
          &mx::metal::CaptureReplay::outputs,
          "The captured output arrays (pinned buffers).");

  metal.def(
      "capture_compiled",
      [](nb::callable fn, nb::args example_inputs) {
        std::vector<mx::array> inputs = tree_flatten(example_inputs, false);
        // Warm / compile: run once and eval so the compiled trace exists and
        // the input leaves are materialized (NOT recorded).
        {
          nb::object warm = fn(*example_inputs);
          std::vector<mx::array> warm_out = tree_flatten(warm, false);
          nb::gil_scoped_release nogil;
          mx::eval(inputs);
          mx::eval(warm_out);
        }
        // Build the graph again and record the steady-state dispatch stream.
        nb::object outs = fn(*example_inputs);
        std::vector<mx::array> outputs = tree_flatten(outs, false);
        mx::metal::capture_begin();
        {
          nb::gil_scoped_release nogil;
          mx::eval(outputs);
        }
        return mx::metal::CaptureReplay::capture(inputs, outputs);
      },
      R"pbdoc(
      EXPERIMENTAL (S2b alpha). Capture the compute dispatch stream of a
      compiled function ``fn`` evaluated on ``example_inputs`` into a
      replayable :class:`CaptureReplay` handle.

      Requires the environment variable ``MLX_CAPTURE_REPLAY`` to be set before
      importing mlx (it makes the Metal Device retain kernel functions so the
      captured pipelines can be rebuilt with indirect-command-buffer support).

      Args:
        fn (Callable): A (typically ``mx.compile``-d) function of the inputs.
        *example_inputs (array): Example array inputs; their buffers are pinned
          and rewritten on replay.
      )pbdoc");

  // --- S4a chained submit ---------------------------------------------------
  // A validated cross-handle blit: copy a SOURCE handle's output leaf into a
  // DESTINATION handle's input leaf, fenced between the two handles' ICBs inside
  // one chained command buffer (the cross-handle generalization of the
  // same-handle CaptureReplayFeedbackPlan). Build it with make_chain_plan.
  nb::class_<mx::metal::ChainPlan>(
      metal,
      "CaptureReplayChainPlan",
      R"pbdoc(
      EXPERIMENTAL (S4a). A validated cross-handle blit (source-handle output leaf
      -> destination-handle input leaf) encoded between the two handles' ICBs in a
      chained command buffer, so a decode stream's output feeds the next stream's
      input ON-DEVICE (no host round-trip). Create it with :func:`make_chain_plan`.
      )pbdoc")
      .def_prop_ro(
          "nbytes",
          [](const mx::metal::ChainPlan& self) { return self.nbytes; },
          "Validated equal byte size of the source and destination leaves.")
      .def_prop_ro(
          "src_out_index",
          [](const mx::metal::ChainPlan& self) { return self.src_out_index; },
          "Output-leaf index of the source handle (blit source).")
      .def_prop_ro(
          "dst_in_index",
          [](const mx::metal::ChainPlan& self) { return self.dst_in_index; },
          "Input-leaf index of the destination handle (blit destination).");

  metal.def(
      "make_chain_plan",
      [](std::shared_ptr<mx::metal::CaptureReplay> src,
         size_t src_out_index,
         std::shared_ptr<mx::metal::CaptureReplay> dst,
         size_t dst_in_index) {
        nb::gil_scoped_release nogil;
        return mx::metal::make_chain_plan(src, src_out_index, dst, dst_in_index);
      },
      "src"_a,
      "src_out_index"_a,
      "dst"_a,
      "dst_in_index"_a,
      R"pbdoc(
      Validate and build a :class:`CaptureReplayChainPlan` copying output leaf
      ``src_out_index`` of ``src`` into input leaf ``dst_in_index`` of ``dst``.
      Both indices must be in range and the two leaves byte-size-equal (raises
      otherwise). Pass the returned plans to :func:`chain_submit` as ``handoffs``.
      )pbdoc");

  metal.def(
      "chain_submit",
      [](std::vector<std::pair<
             std::shared_ptr<mx::metal::CaptureReplay>,
             std::shared_ptr<mx::metal::CaptureReplay::FeedbackPlan>>> stages,
         std::vector<std::shared_ptr<mx::metal::ChainPlan>> handoffs) {
        nb::gil_scoped_release nogil;
        std::vector<mx::metal::ChainStage> chain_stages;
        chain_stages.reserve(stages.size());
        for (auto& s : stages) {
          chain_stages.push_back(mx::metal::ChainStage{s.first, s.second});
        }
        return mx::metal::chain_submit(chain_stages, handoffs);
      },
      "stages"_a,
      "handoffs"_a = std::vector<std::shared_ptr<mx::metal::ChainPlan>>{},
      R"pbdoc(
      Encode several captured handles' ICBs into ONE command buffer and commit it
      with a SINGLE host submit (collapsing the per-stream submit/wait round-trips
      of a serving cycle). ``stages`` is an ordered list of
      ``(handle, feedback_plan_or_None)`` tuples — each handle's ICB, then its
      same-handle feedback blits — and ``handoffs`` is a list of
      :class:`CaptureReplayChainPlan` cross-handle blits inserted between the
      stages (each handoff's source stage must precede its destination stage).
      Every encoder is serialized by a fresh MTLFence per boundary, so producers
      are ordered before consumers. Host input rewrites stay host-side per handle
      BEFORE the call. Returns an integer ticket for :func:`chain_wait`. Serial-use
      only: one chained command buffer in flight.
      )pbdoc");

  metal.def(
      "chain_wait",
      [](uint64_t ticket) {
        nb::gil_scoped_release nogil;
        mx::metal::chain_wait(ticket);
      },
      "ticket"_a,
      R"pbdoc(
      Block until the chained command buffer for ``ticket`` completes (the single
      per-cycle wait). After it returns, read the small decision outputs from the
      stage handles' :meth:`CaptureReplay.read_output`; the advancing decode state
      was fed on-device by the feedback / handoff blits.
      )pbdoc");
}
