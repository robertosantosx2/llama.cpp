#pragma once

// Shared, opt-in NEPENTHES event contract for llama.cpp and ggml backends.
#include "ggml.h"
#include "ggml-nepenthes-c.h"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace nepenthes {

enum class EventType : std::uint8_t {
    request_start, request_end, graph_compute, dispatch,
    model_load, model_unload,
    routing, cache_hit, cache_miss, eviction, prefetch,
    kv_cache, allocation, free_event, transfer, kernel,
    sync, cpu_work, stall, oom, fallback, error, clock_calibration, nvtx_marker,
};

enum EventFlags : std::uint16_t {
    phase_instant = 0,
    phase_begin = 1u << 0,
    phase_end = 1u << 1,
    flag_failed = 1u << 2,
    flag_host = 1u << 3,
    flag_device = 1u << 4,
};

enum class Backend : std::uint8_t { unknown, cpu, cuda, metal, vulkan, rocm, musa };
enum class MemoryKind : std::uint8_t { unknown, host, host_pinned, device, managed, vmm };
enum class MemcpyKind : std::uint8_t {
    unknown, host_to_host, host_to_device, device_to_host, device_to_device, peer_to_peer, default_kind
};
enum class SyncType : std::uint8_t { unknown, stream, device, event, context };
enum class KvOp : std::uint8_t {
    unknown, clear, seq_rm, seq_cp, seq_keep, seq_add, seq_div, prepare, update, apply_ubatch
};
enum class AttributionLevel : std::uint8_t { unknown, exact, windowed, device_wide, estimated };

struct EventMetadata {
    Backend backend = Backend::unknown;
    MemoryKind memory_kind = MemoryKind::unknown;
    MemcpyKind memcpy_kind = MemcpyKind::unknown;
    SyncType sync_type = SyncType::unknown;
    KvOp kv_op = KvOp::unknown;
    AttributionLevel attribution_level = AttributionLevel::unknown;
    std::uint32_t raw_memcpy_kind = 0xffffffffu;
};

struct Event {
    std::uint64_t monotonic_ns = 0;
    std::uint64_t timestamp_unix_ns = 0;
    std::uint64_t duration_ns = 0;
    std::uint64_t request_id = 0;
    std::uint64_t correlation_token = 0;
    std::uint64_t value0 = 0;
    std::uint64_t value1 = 0;
    EventType type = EventType::error;
    std::uint16_t flags = phase_instant;
    EventMetadata metadata{};
    const char * label = nullptr; // borrowed; sinks must consume/copy during callback
};

using Sink = void (*)(const Event &) noexcept;

// These functions are implemented once in ggml-base so llama and backend
// translation units share one callback and one in-flight emitter counter.
GGML_API void set_sink(Sink sink) noexcept;
GGML_API void emit(EventType type, const char * label = nullptr,
                   std::uint64_t value0 = 0, std::uint64_t value1 = 0,
                   std::uint16_t flags = phase_instant,
                   std::uint64_t request_id = 0,
                   std::uint64_t duration_ns = 0,
                   std::uint64_t correlation_token = 0,
                   EventMetadata metadata = {}) noexcept;

inline Backend current_gpu_backend() noexcept {
#if defined(GGML_USE_HIP)
    return Backend::rocm;
#elif defined(GGML_USE_MUSA)
    return Backend::musa;
#else
    return Backend::cuda;
#endif
}

inline EventMetadata gpu_metadata(MemoryKind memory = MemoryKind::unknown,
                                  MemcpyKind copy = MemcpyKind::unknown,
                                  SyncType sync = SyncType::unknown,
                                  std::uint32_t raw_copy = 0xffffffffu) noexcept {
    EventMetadata metadata;
    metadata.backend = current_gpu_backend();
    metadata.memory_kind = memory;
    metadata.memcpy_kind = copy;
    metadata.sync_type = sync;
    metadata.raw_memcpy_kind = raw_copy;
    return metadata;
}

inline EventMetadata cpu_metadata() noexcept {
    EventMetadata metadata;
    metadata.backend = Backend::cpu;
    return metadata;
}

inline EventMetadata kv_metadata(KvOp op) noexcept {
    EventMetadata metadata;
    metadata.kv_op = op;
    return metadata;
}

inline EventMetadata context_sync_metadata() noexcept {
    EventMetadata metadata;
    metadata.sync_type = SyncType::context;
    return metadata;
}

inline void emit_gpu(EventType type, const char * label = nullptr,
                      std::uint64_t value0 = 0, std::uint64_t value1 = 0,
                      std::uint16_t flags = phase_instant,
                      std::uint64_t request_id = 0,
                      std::uint64_t duration_ns = 0,
                      std::uint64_t correlation_token = 0,
                      EventMetadata metadata = {}) noexcept {
    metadata.backend = current_gpu_backend();
    emit(type, label, value0, value1, flags, request_id, duration_ns, correlation_token, metadata);
}

inline std::uint64_t next_correlation_token() noexcept {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

inline std::uint64_t now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

class ScopedEvent {
public:
    ScopedEvent(EventType type, const char * label,
                std::uint64_t value0 = 0, std::uint64_t value1 = 0,
                std::uint64_t request_id = 0,
                EventMetadata metadata = {}) noexcept
        : type_(type), label_(label), value0_(value0), value1_(value1),
          request_id_(request_id), metadata_(metadata), start_ns_(now_ns()) {
        emit(type_, label_, value0_, value1_, phase_begin, request_id_, 0, 0, metadata_);
    }

    ScopedEvent(const ScopedEvent &) = delete;
    ScopedEvent & operator=(const ScopedEvent &) = delete;

    ~ScopedEvent() noexcept {
        const std::uint64_t end_ns = now_ns();
        emit(type_, label_, value0_, value1_, phase_end, request_id_,
             end_ns >= start_ns_ ? end_ns - start_ns_ : 0, 0, metadata_);
    }

private:
    EventType type_;
    const char * label_;
    std::uint64_t value0_;
    std::uint64_t value1_;
    std::uint64_t request_id_;
    EventMetadata metadata_{};
    std::uint64_t start_ns_;
};

} // namespace nepenthes

