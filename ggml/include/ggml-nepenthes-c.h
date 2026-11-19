#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// C ABI used by the CPU backend without depending on C++ trace types.
GGML_API int ggml_nepenthes_trace_enabled(void);
GGML_API void ggml_nepenthes_emit_cpu_work(uint64_t worker_id, uint64_t node_count, uint64_t duration_ns);
GGML_API void ggml_nepenthes_emit_cpu_wait(uint64_t worker_id, uint64_t duration_ns);

#ifdef __cplusplus
}
#endif
