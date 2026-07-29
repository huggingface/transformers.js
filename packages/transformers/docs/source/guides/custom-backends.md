# Custom inference backends

Transformers.js can load curated model backends that use a runtime other than the default ONNX provider. A pipeline-facing backend supplies a fixed model ID and a `load()` function:

```ts
import type {
  CausalGenerationCapabilitiesV1,
  InferenceBackend,
  InferenceModel,
  PlanAutoregressiveSessionV1,
} from '@huggingface/transformers';

const causalGeneration = {
  sessionVersion: 1,
  maxBatchSize: 1,
  cpuLogits: false,
  cpuModes: [],
  planModes: ['greedy'],
  declarativePlans: ['argmax'],
  tokenPipeline: { defaultDepth: 4, maxDepth: 4 },
} as const satisfies CausalGenerationCapabilitiesV1;

export const backend: InferenceBackend = {
  modelId: 'organization/curated-model',
  capabilities: {
    devices: ['webgpu'],
    dtypes: ['auto'],
    tasks: ['text-generation'],
  },
  async load(options): Promise<InferenceModel> {
    // Load weights and create runtime-owned model state here.
    return {
      capabilities: { causalGeneration },
      async createAutoregressiveSession(sessionOptions): Promise<PlanAutoregressiveSessionV1> {
        throw new Error('Example only');
      },
      async dispose() {},
    };
  },
};
```

Pass the backend anywhere a model ID is accepted:

```js
const generator = await pipeline('text-generation', backend, {
  device: 'webgpu',
  dtype: 'auto',
  signal,
  artifactProvider,
});
```

## Capabilities

Static backend capabilities are advisory and allow `pipeline()` to reject unsupported tasks early. The loaded model's `capabilities` are authoritative. Execution families are optional and task-specific: decoder-only generation uses `causalGeneration`; forward-only, encoder-decoder, diffusion, and audio runtimes should not imitate the causal protocol.

Plan-only causal runtimes may declare `cpuLogits: false`, an empty `cpuModes`, and a supported greedy plan. Requests requiring JavaScript logits processors, sampling, or returned scores cannot use that plan and fail before a session is created. Pull sessions expose CPU logits leases for full Transformers.js policy compatibility.

## Session lifecycle

Transformers.js owns generation policy, stopping, callbacks, streaming, and final output construction. The runtime owns inference state and the KV cache.

- `generateWithPlan()` yields ordered token decisions. Transformers.js awaits the iterator's `return()` when generation stops early.
- A logits lease must remain valid until its synchronous, idempotent `release()` is called.
- Session `dispose()` is always awaited after generation.
- `sessionConcurrency.maxActiveSessions` is enforced fail-fast. Callers that want queueing must serialize generation themselves.
- Abort signals are forwarded through loading and session creation. Backends must release partially created resources before propagating cancellation.

Protocol V1 does not export runtime KV state. With `return_dict_in_generate: true`, custom sessions return an empty `DynamicCache` for result-shape compatibility and dispose their runtime-owned cache with the session.

## Artifact providers

An `InferenceArtifactProvider` can supply JSON and random-access byte sources. Byte ranges are half-open (`[begin, end)`), independent reads may complete out of order, and returned arrays must be owned by the caller. `close()` is idempotent, rejects new reads, and waits for existing reads without implicitly aborting them.

An explicit artifact provider takes precedence inside the custom backend. Provider failures must be propagated after cleanup rather than retried through another transport.
