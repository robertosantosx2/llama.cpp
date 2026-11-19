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

- Model lifecycle: `model_load` spans tensor setup and weight loading; `model_unload` records model byte/element metadata at destruction. The sink is initialized from the model-loader constructor so allocation/loading events are not missed.


## CPU worker probes (follow-up slice)

The CPU backend now emits two additional event classes through a small C ABI bridge into the shared ggml-base sink:

- `cpu_work` / `cpu.worker_graph_compute`: one span per CPU graph worker invocation, with worker index, graph node count and elapsed host duration. The duration includes that worker's graph loop and its graph barriers; it is not an exclusive CPU-time counter.
- `stall` / `cpu.worker_wait`: elapsed time spent in the condition-variable sleep path while an idle worker waits for new graph work. Busy polling is not counted as sleeping wait time.

When tracing is disabled, the worker checks the shared enabled flag and skips the clock reads and event bridge. These events use request_id=0 because the CPU worker pool has no safe request-scoped identity propagation at this layer.

No generic CPU expert-prefetch worker is present in this upstream llama.cpp path; NEPENTHES does not fabricate prefetch events. Model-specific MoE CPU fallback scheduling remains reported by the MoE planner events. This slice does not instrument every CPU operator or every barrier individually, avoiding a high-frequency event flood.


## Explicit request correlation and clock calibration (follow-up)

The server assigns a process-monotonic `correlation_token` to each accepted slot task and emits the same token on its `request_start` and `request_end` records. The native server `request_id` remains unchanged. Tokens are opaque unsigned 64-bit values, unique within one process lifetime; they are not UUIDs and must be scoped by `run_id`.

The JSONL sink emits `clock_calibration` records at sink start and shutdown. Those records, plus request records carrying a correlation token, capture `monotonic_ns` and `timestamp_unix_ns` at the producer callsite, not when the writer thread later drains the queue. Other high-frequency events carry the monotonic clock only and use `timestamp_unix_ns=0` to indicate that no wall-clock sample was taken. This avoids a system-clock read on every hot-path event. `clock_domain` is `steady_clock`. The importer must retain both values and estimate offset/drift from start/end calibration records; these host-only records do not calibrate the Nsight clock.

`attribution_level=exact` on a record means that record itself carries the explicit request token. Events without a token emit null attribution. This slice deliberately does **not** propagate one request token into shared graph/CPU/GPU work: llama.cpp can batch multiple slots into one graph, so assigning one slot token to that graph would be false attribution. NVTX/CUPTI external-correlation markers and verified Nsight round-trip remain C07 work; no kernel attribution is claimed yet.
