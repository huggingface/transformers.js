# Phi-4-mini decode graph capture benchmark

Measured on 2026-09-06 using the Transformers.js generation path, ORT Web's
Asyncify runtime, and Dawn's Node WebGPU binding on an NVIDIA GeForce RTX 5080
(D3D12 driver 32.0.16.1656), Windows, Node.js v24.20.0.

**Runtime prerequisite:** these measurements include
[ONNX Runtime #32456](https://github.com/microsoft/onnxruntime/pull/32456).
The currently pinned ORT package ignores the extra capture provider setting
without that fix. Both comparison modes use the same patched JS runtime and
the same unmodified WASM binaries and model assets.

## Results

| Metric (median of five measured runs) | Capture off | Decode capture on |
| --- | ---: | ---: |
| Decode throughput | 55.32 tokens/s | 130.76 tokens/s |
| Time to first token | 38.93 ms | 40.25 ms |

Decode throughput improved **2.36× (+136.4%)**. An earlier independent series
measured 51.46 → 128.45 tokens/s (**2.50×**). Prefill remains uncaptured.
All warmup and measured runs generated identical token IDs across modes.

The final benchmark automatically observed two native
`Replaying the captured WebGpuExecutionProvider graph` log events in a separate,
untimed verification pass. Logging was disabled for the measured sessions.
The script refuses to report a speedup if it does not observe actual replay.

## Workload and method

- Model: Phi-4-mini-instruct (3.8B), ORT GenAI graph-ready export with GQA,
  fused rotary embedding, INT4 linear weights, and FP16 activations/cache.
  The external weight file contains 2,491,416,576 bytes.
- Batch size 1; 128 prompt tokens and exactly 128 generated tokens; greedy
  sampling; fixed KV capacity of 2048 tokens; WebGPU validation mode `basic`.
- Two warmup generations per mode, then five measured generations per mode.
  Capture-off runs precede capture-on runs; the models are loaded sequentially.
- Both modes use the same static, shared KV input/output buffers. Only capture
  is toggled. Prefill uses `gpu_graph_id=-1`; captured decode uses ID `0`.
- Decode throughput is `127 / (last_token_time - first_token_time)` and includes
  Transformers.js generation overhead and sampling. Model loading, prompt
  tokenization, prefill, verification, and warmup are excluded from this interval.
- TTFT is measured from `generate()` to its first generated-token callback.

These are ORT **Web** measurements through a Node WebGPU binding, not native
ORT GenAI or browser measurements. They do not establish performance for other
GPUs, context capacities, model exports, or browser versions. Run-to-run variation
is retained in the [raw results](./graph_capture_benchmark.json), alongside asset
hashes, token IDs, and both measurement series.

## Reproduce

From the repository root, with Node.js 24 or newer and a Dawn Node binding built
with DXC (`shader-f16` and `subgroups` must be available):

```sh
pnpm install --frozen-lockfile

node packages/transformers/scripts/benchmarks/build-ort-extra-options.mjs \
  --output .cache/ort.webgpu.extra-options.mjs

node packages/transformers/scripts/benchmarks/graph-capture.mjs \
  --model /path/to/graph-ready/Phi-4-mini-instruct \
  --webgpu-module /path/to/dawn.node \
  --ort-module .cache/ort.webgpu.extra-options.mjs \
  --prompt-tokens 128 --new-tokens 128 --context 2048 \
  --warmup 2 --runs 5 --output graph-capture-results.json
```

The input directory must contain `genai_config.json`, `model.onnx`,
`model.onnx.data`, and tokenizer assets. The export was built with
`enable_webgpu_graph=true`, shared embeddings, and a pruned LM head.
The script stages Transformers.js configuration separately and leaves the
original model directory unchanged.
