#include "nepenthes_jsonl_sink.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>

namespace nepenthes {

void ensure_sink_from_env() noexcept {
    static std::unique_ptr<JsonlSink> instance = []() -> std::unique_ptr<JsonlSink> {
        const char * path = std::getenv("NEPENTHES_TRACE_JSONL");
        if (path == nullptr || *path == '\0') {
            return nullptr;
        }
        const char * run = std::getenv("NEPENTHES_RUN_ID");
        const char * profile = std::getenv("NEPENTHES_PROFILE");
        const bool request_only = profile != nullptr && std::string(profile) == "latency-critical";
        try {
            return std::make_unique<JsonlSink>(path, run ? run : "llama.cpp", request_only);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "NEPENTHES: cannot start JSONL sink: %s\n", e.what());
            return nullptr;
        }
    }();
    (void) instance;
}

} // namespace nepenthes
