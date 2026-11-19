# NEPENTHES runtime trace in llama.cpp

## Activation

Tracing is disabled by default. Set the output path before starting llama-server:

```bash
export NEPENTHES_TRACE_JSONL="$HOME/nepenthes-run.jsonl"
export NEPENTHES_RUN_ID="qwen35-$(date +%Y%m%d-%H%M%S)"
./build/bin/llama-server --model /path/to/model.gguf
```

The sink is initialized lazily when the server accepts its first task or the first graph is computed. If NEPENTHES_TRACE_JSONL is unset or empty, no writer thread or output file is created. The run ID defaults to `llama.cpp`.

## Event contract

Each event is one JSON object per line. Current fields:

- `schema_version`: event schema version (currently `1.0`).
- `run_id`, `event_id`, `source`: run and producer identity.
- `timestamp_unix_ns`: wall-clock timestamp for cross-tool correlation.
- `monotonic_ns`: steady-clock timestamp for local duration/order analysis.
- `request_id`: actual server task ID for request lifecycle events; zero for internal events without safe correlation.
- `event_kind`, `label`: event class and stable callsite label.
- `value0`, `value1`: callsite-specific integer metadata.
- `duration_ns`: duration for scoped begin/end events; zero for instant events.
- `flags`: phase and host/device metadata.

A final `sink_summary` record reports events drained, dropped events and writer errors. Queue overflow is explicit rather than silently treated as complete telemetry.

## Current probe coverage

| Area | Events | Meaning |
|---|---|---|
| Server request lifecycle | `request_start`, `request_end` | Server task ID and slot ID; request_id is the task ID |
| Graph execution | `graph_compute` | Host-side graph-compute call duration, graph node count and batched flag |
| Synchronization | `sync` | Context synchronization duration |
| MoE expert cache | `cache_hit`, `cache_miss`, `eviction` | Planner hit/miss counts, CPU fallback count, evicted layer/expert |
| MoE expert uploads | `transfer` | Bytes submitted for expert weight uploads; host-side async submission, not GPU completion time |
| KV cache | `kv_cache` | clear, seq_rm, seq_cp, seq_keep, seq_add, seq_div, prepare, update and apply_ubatch durations/metadata |

## Concurrency and overhead model

- Producers submit fixed-size events to a bounded 8192-entry multi-producer queue.
- Producers do not open, format or write files. A dedicated writer thread owns JSONL serialization and file I/O.
- Event labels are borrowed static strings; no label allocation is performed on the emit path.
- Sink shutdown disables new callbacks, waits for callbacks already in flight, drains the queue, writes a summary record and joins the writer.
- The queue may drop events under sustained overload; the summary record exposes the drop count.
- The sink appends to the configured path. Use a unique run ID and preferably a fresh file per run.

## Interpretation limits

- Graph-compute spans are not individual kernel timings.
- The MoE upload event measures bytes submitted at the host callsite, not DMA completion.
- Internal KV and MoE events use request_id=0. No cross-thread request identity is fabricated.
- Prompts, generated text, tokens and model input/output payloads are not recorded.
- General allocator/OOM/fallback probes, CUDA memcpy callsites, CUDA op dispatch, and additional CPU worker/prefetch spans are follow-up work. Actual GPU kernel timings and runtime transfer activities remain sourced from Nsight Systems/CUPTI.

## Validation

The NEPENTHES workflow compiles and runs the event and JSONL tests, then builds the CPU llama-server target. This validates the CPU code path only. CUDA compilation and a reproducible runtime capture on the target MSI are separate acceptance gates.


## Backend and CUDA probes (follow-up slice)

The shared event API now lives in ggml-base so llama and backend translation units use one process-wide sink. The added backend coverage is:

- Generic ggml backend buffer creation and destruction (wrapper lifetime and byte size).
- CUDA device allocation outcomes: success, OOM and other errors.
- Legacy CUDA pool reuse, logical return, actual release, and OOM flush/retry.
- CUDA VMM pool allocation/free and physical allocation outcomes.
- Pinned-host allocation/free and fallback to ordinary CPU buffers.
- CUDA H2D/D2H transfers through synchronous buffer operations and asynchronous backend submissions; 2D and batch paths retain row/count or tensor-count metadata.
- CUDA D2D/peer copy attempts and batch copies.
- CUDA op dispatch duration, identified by ggml op enum and output element count.
- CUDA stream synchronization duration.

Interpretation is deliberately layered: a generic ggml buffer event is not the same as a physical CUDA allocation. Pool-return/free events distinguish logical reuse from a physical cudaFree/unmap. Async transfer events measure host submission, while the synchronous buffer-interface events include the explicit stream synchronization in those callsites. CUDA op dispatch is host dispatch time, not GPU kernel duration.

The CUDA CI job compiles the backend for compute capability 8.6 using a CUDA development container. It does not require a GPU and does not validate runtime behavior on the MSI. Nsight Systems/CUPTI remain the source for device-side kernel timing and transfer completion.
