#include "nepenthes_jsonl_sink.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

int main() {
    const std::string path = "nepenthes-jsonl-test.jsonl";
    std::remove(path.c_str());
    {
        nepenthes::JsonlSink sink(path, "run\"line\nbreak");
        for (std::uint64_t i = 0; i < 256; ++i) {
            nepenthes::emit_cuda(nepenthes::EventType::kernel, "cuda.kernel", i, 4096,
                            nepenthes::phase_instant, 0, 0, i + 1,
                            nepenthes::cuda_metadata(nepenthes::MemoryKind::device));
        }
        nepenthes::emit(nepenthes::EventType::nvtx_marker, "request.start", 100, 120,
                        nepenthes::phase_begin, 42, 0, 999);
        nepenthes::emit(nepenthes::EventType::kv_cache, "kv.seq_rm", 1, 0,
                        nepenthes::phase_instant, 0, 0, 0,
                        nepenthes::kv_metadata(nepenthes::KvOp::seq_rm));
        std::vector<std::thread> producers;
        for (std::uint64_t t = 0; t < 4; ++t) {
            producers.emplace_back([t] {
                for (std::uint64_t i = 0; i < 1000; ++i) {
                    nepenthes::emit_cuda(nepenthes::EventType::transfer, "moe.transfer", t, 1024,
                                    nepenthes::phase_instant, t + 1, 0, 0,
                                    nepenthes::cuda_metadata(nepenthes::MemoryKind::device,
                                            nepenthes::MemcpyKind::host_to_device,
                                            nepenthes::SyncType::stream, 1));
                }
            });
        }
        for (auto & producer : producers) producer.join();
        sink.close();
        sink.close();
        assert(sink.dropped() == 0);
        assert(sink.write_errors() == 0);
    }

    std::ifstream input(path);
    assert(input.good());
    std::string line;
    std::size_t count = 0;
    bool summary_seen = false;
    while (std::getline(input, line)) {
        // Every record must occupy one physical line and end with its JSON brace.
        // This catches a literal backslash+n being written instead of a newline.
        assert(!line.empty());
        assert(line.back() == '}');
        assert(line.find("\"source\":\"llama.cpp\"") != std::string::npos);
        assert(line.find("\"run_id\":\"run\\\"line\\nbreak\"") != std::string::npos);
        assert(line.find("\"schema_version\":\"1.1\"") != std::string::npos);
        if (line.find("\"event_kind\":\"kernel\"") != std::string::npos) {
            assert(line.find("\"correlation_token\":") != std::string::npos);
            assert(line.find("\"attribution_level\":\"exact\"") != std::string::npos);
            assert(line.find("\"clock_domain\":\"steady_clock\"") != std::string::npos);
            assert(line.find("\"backend\":\"cuda\"") != std::string::npos);
            assert(line.find("\"memory_kind\":\"device\"") != std::string::npos);
        }
        if (line.find("\"event_kind\":\"transfer\"") != std::string::npos) {
            assert(line.find("\"memcpy_kind\":\"host_to_device\"") != std::string::npos);
            assert(line.find("\"raw_memcpy_kind\":1") != std::string::npos);
            assert(line.find("\"sync_type\":\"stream\"") != std::string::npos);
        }
        if (line.find("\"event_kind\":\"kv_cache\"") != std::string::npos) {
            assert(line.find("\"kv_op\":\"seq_rm\"") != std::string::npos);
        }
        if (line.find("\"event_kind\":\"nvtx_marker\"") != std::string::npos) {
            assert(line.find("\"correlation_token\":999") != std::string::npos);
            assert(line.find("\"attribution_level\":\"exact\"") != std::string::npos);
        }
        if (line.find("\"event_kind\":\"clock_calibration\"") != std::string::npos) {
            assert(line.find("\"timestamp_unix_ns\":") != std::string::npos);
            assert(line.find("\"monotonic_ns\":") != std::string::npos);
        }
        if (line.find("\"event_kind\":\"sink_summary\"") != std::string::npos) {
            summary_seen = true;
            assert(line.find("\"events_drained\":4260") != std::string::npos);
            assert(line.find("\"dropped_events\":0") != std::string::npos);
            assert(line.find("\"write_errors\":0") != std::string::npos);
        }
        ++count;
    }
    input.close();
    std::remove(path.c_str());
    assert(count == 4261);
    assert(summary_seen);
    return 0;
}
