#include "nepenthes_jsonl_sink.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    const bool keep_trace = argc > 1 && std::string(argv[1]) == "--keep-trace";
    const std::string path = "nepenthes-jsonl-test.jsonl";
    std::remove(path.c_str());
    {
        nepenthes::JsonlSink sink(path, "run\"line\nbreak");
        for (std::uint64_t i = 0; i < 256; ++i) {
            nepenthes::emit(nepenthes::EventType::kernel, "cuda.kernel", i, 4096,
                            nepenthes::phase_instant, 0, 0, i + 1);
        }
        nepenthes::emit(nepenthes::EventType::request_start, "server.task", 7, 2,
                        nepenthes::phase_instant, 42, 0, 999);
        nepenthes::emit(nepenthes::EventType::nvtx_marker, "request.start", 100, 120,
                        nepenthes::phase_begin, 42, 0, 999);
        nepenthes::emit(nepenthes::EventType::request_end, "server.task", 7, 2,
                        nepenthes::phase_instant, 42, 0, 999);
        std::vector<std::thread> producers;
        for (std::uint64_t t = 0; t < 4; ++t) {
            producers.emplace_back([t] {
                for (std::uint64_t i = 0; i < 1000; ++i) {
                    nepenthes::emit(nepenthes::EventType::transfer, "moe.transfer", t, 1024,
                                    nepenthes::phase_instant, t + 1);
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
        if (line.find("\"event_kind\":\"kernel\"") != std::string::npos) {
            assert(line.find("\"correlation_token\":") != std::string::npos);
            assert(line.find("\"attribution_level\":\"exact\"") != std::string::npos);
            assert(line.find("\"clock_domain\":\"steady_clock\"") != std::string::npos);
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
            assert(line.find("\"events_drained\":4261") != std::string::npos);
            assert(line.find("\"dropped_events\":0") != std::string::npos);
            assert(line.find("\"write_errors\":0") != std::string::npos);
        }
        ++count;
    }
    input.close();
    if (!keep_trace) std::remove(path.c_str());
    assert(count == 4262);
    assert(summary_seen);

    // Existing trace paths must be rejected without truncating or appending to prior evidence.
    const std::string existing_path = "nepenthes-jsonl-existing.jsonl";
    std::remove(existing_path.c_str());
    {
        std::ofstream existing(existing_path);
        existing << "preserve-this-run\n";
    }
    bool existing_rejected = false;
    try {
        nepenthes::JsonlSink existing_sink(existing_path, "must-not-append");
    } catch (const std::runtime_error &) {
        existing_rejected = true;
    }
    assert(existing_rejected);
    {
        std::ifstream existing(existing_path);
        std::getline(existing, line);
        assert(line == "preserve-this-run");
    }
    std::remove(existing_path.c_str());

    // latency-critical retains only request lifecycle and clock calibration events.
    const std::string latency_path = "nepenthes-jsonl-latency-critical.jsonl";
    std::remove(latency_path.c_str());
    {
        nepenthes::JsonlSink sink(latency_path, "latency-critical", true);
        nepenthes::emit(nepenthes::EventType::kernel, "cuda.kernel", 1, 2);
        nepenthes::emit(nepenthes::EventType::transfer, "cuda.transfer", 1, 2);
        nepenthes::emit(nepenthes::EventType::request_start, "request.start", 0, 0,
                        nepenthes::phase_instant, 42, 0, 77);
        nepenthes::emit(nepenthes::EventType::request_end, "request.end", 0, 0,
                        nepenthes::phase_instant, 42, 0, 77);
        sink.close();
        assert(sink.dropped() == 0);
        assert(sink.write_errors() == 0);
    }
    std::ifstream latency_input(latency_path);
    assert(latency_input.good());
    std::size_t latency_count = 0;
    bool request_start_seen = false;
    bool request_end_seen = false;
    bool kernel_seen = false;
    bool transfer_seen = false;
    std::size_t calibration_count = 0;
    while (std::getline(latency_input, line)) {
        assert(!line.empty() && line.back() == '}');
        request_start_seen |= line.find("\"event_kind\":\"request_start\"") != std::string::npos;
        request_end_seen |= line.find("\"event_kind\":\"request_end\"") != std::string::npos;
        kernel_seen |= line.find("\"event_kind\":\"kernel\"") != std::string::npos;
        transfer_seen |= line.find("\"event_kind\":\"transfer\"") != std::string::npos;
        calibration_count += line.find("\"event_kind\":\"clock_calibration\"") != std::string::npos;
        ++latency_count;
    }
    latency_input.close();
    std::remove(latency_path.c_str());
    assert(request_start_seen && request_end_seen);
    assert(!kernel_seen && !transfer_seen);
    assert(calibration_count == 2);
    assert(latency_count == 5); // 4 accepted events + sink summary
    return 0;
}
