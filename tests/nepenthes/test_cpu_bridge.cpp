#include "ggml-nepenthes.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

extern "C" int ggml_nepenthes_trace_enabled(void);
extern "C" void ggml_nepenthes_emit_cpu_work(std::uint64_t worker_id,
                                               std::uint64_t node_count,
                                               std::uint64_t duration_ns);
extern "C" void ggml_nepenthes_emit_cpu_wait(std::uint64_t worker_id,
                                               std::uint64_t duration_ns);

static nepenthes::Event events[2];
static std::size_t event_count = 0;

static void collect(const nepenthes::Event & event) noexcept {
    assert(event_count < 2);
    events[event_count++] = event;
}

int main() {
    using namespace nepenthes;
    assert(ggml_nepenthes_trace_enabled() == 0);
    assert(event_count == 0);

    set_sink(&collect);
    assert(ggml_nepenthes_trace_enabled() == 1);
    ggml_nepenthes_emit_cpu_work(3, 128, 9000);
    ggml_nepenthes_emit_cpu_wait(2, 7000);

    assert(event_count == 2);
    assert(events[0].type == EventType::cpu_work);
    assert(events[0].label != nullptr);
    assert(events[0].value0 == 3);
    assert(events[0].value1 == 128);
    assert(events[0].duration_ns == 9000);
    assert(events[1].type == EventType::stall);
    assert(events[1].label != nullptr);
    assert(events[1].value0 == 2);
    assert(events[1].duration_ns == 7000);

    set_sink(nullptr);
    assert(ggml_nepenthes_trace_enabled() == 0);
    return 0;
}
