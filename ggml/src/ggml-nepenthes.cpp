#include "ggml-nepenthes.hpp"

#include <atomic>
#include <thread>

namespace {

std::atomic<nepenthes::Sink> & sink_slot() noexcept {
    static std::atomic<nepenthes::Sink> slot{nullptr};
    return slot;
}

std::atomic<std::uint32_t> & active_emitters() noexcept {
    static std::atomic<std::uint32_t> count{0};
    return count;
}

} // namespace

namespace nepenthes {

void set_sink(Sink sink) noexcept {
    sink_slot().store(sink, std::memory_order_release);
    if (sink == nullptr) {
        while (active_emitters().load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }
}

void emit(EventType type, const char * label,
          std::uint64_t value0, std::uint64_t value1,
          std::uint16_t flags, std::uint64_t request_id,
          std::uint64_t duration_ns, std::uint64_t correlation_token) noexcept {
    const Sink sink = sink_slot().load(std::memory_order_acquire);
    if (sink == nullptr) {
        return;
    }

    active_emitters().fetch_add(1, std::memory_order_acq_rel);
    // An emitter may have loaded the old callback immediately before shutdown.
    if (sink_slot().load(std::memory_order_acquire) != sink) {
        active_emitters().fetch_sub(1, std::memory_order_release);
        return;
    }

    Event event;
    event.monotonic_ns = now_ns();
    event.timestamp_unix_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    event.duration_ns = duration_ns;
    event.request_id = request_id;
    event.correlation_token = correlation_token;
    event.value0 = value0;
    event.value1 = value1;
    event.type = type;
    event.flags = flags;
    event.label = label;
    sink(event);

    active_emitters().fetch_sub(1, std::memory_order_release);
}

} // namespace nepenthes


int ggml_nepenthes_trace_enabled(void) {
    return sink_slot().load(std::memory_order_acquire) != nullptr ? 1 : 0;
}

void ggml_nepenthes_emit_cpu_work(std::uint64_t worker_id,
                                               std::uint64_t node_count,
                                               std::uint64_t duration_ns) {
    nepenthes::emit(nepenthes::EventType::cpu_work, "cpu.worker_graph_compute",
            worker_id, node_count, nepenthes::phase_instant, 0, duration_ns);
}

void ggml_nepenthes_emit_cpu_wait(std::uint64_t worker_id,
                                               std::uint64_t duration_ns) {
    nepenthes::emit(nepenthes::EventType::stall, "cpu.worker_wait",
            worker_id, 0, nepenthes::phase_instant, 0, duration_ns);
}
