#include "nepenthes_trace.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>
#include <chrono>

static std::uint64_t seen = 0;
static std::uint64_t last_duration = 0;
static std::uint16_t last_flags = 0;
static std::uint64_t last_correlation_token = 0;
static std::uint64_t last_wall_ns = 0;
static std::atomic<bool> callback_entered{false};
static std::atomic<bool> release_callback{false};

static void collect(const nepenthes::Event & event) noexcept {
    ++seen;
    last_duration = event.duration_ns;
    last_flags = event.flags;
    last_correlation_token = event.correlation_token;
    last_wall_ns = event.timestamp_unix_ns;
}

static void blocking_collect(const nepenthes::Event &) noexcept {
    callback_entered.store(true, std::memory_order_release);
    while (!release_callback.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

int main() {
    using namespace nepenthes;

    emit(EventType::request_start, "disabled");
    assert(seen == 0);

    set_sink(&collect);
    emit(EventType::cache_hit, "expert", 7, 4096, flag_device, 42, 0, 777);
    assert(seen == 1);
    assert(last_flags == flag_device);
    assert(last_correlation_token == 777);
    assert(last_wall_ns > 0);
    const auto token_a = next_correlation_token();
    const auto token_b = next_correlation_token();
    assert(token_a != 0 && token_b == token_a + 1);

    {
        ScopedEvent phase(EventType::sync, "worker_wait", 3, 0, 42);
    }
    assert(seen == 3);
    assert(last_flags == phase_end);
    (void) last_duration;

    set_sink(nullptr);
    emit(EventType::request_end, "disabled");
    assert(seen == 3);

    // Shutdown must wait for a callback that has already entered the sink.
    set_sink(&blocking_collect);
    std::thread producer([] {
        emit(EventType::kernel, "shutdown-race");
    });
    while (!callback_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    std::atomic<bool> disable_done{false};
    std::thread disabler([&] {
        set_sink(nullptr);
        disable_done.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!disable_done.load(std::memory_order_acquire));

    release_callback.store(true, std::memory_order_release);
    producer.join();
    disabler.join();
    assert(disable_done.load(std::memory_order_acquire));
}
