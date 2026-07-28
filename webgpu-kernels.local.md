# Custom inference backends in Transformers.js

## Generation update

The original model-level backend decision remains active, but the generation portion of this document is superseded by the runtime-reviewed V1 protocol in `webgpu-compat.local.md`.

Custom generation models now expose `generationCapabilities` and `createAutoregressiveSession()`. Transformers.js installs the public `generate()`, owns generation policy and finalization, and drives either leased CPU logits or an approved declarative runtime plan. Custom runtimes should not implement public generation policy themselves.

The initial artifact-loading agreement is also finalized there: an injected random-access provider takes precedence, while the first Gemma4 adapter may otherwise use runtime IO and must reject unsupported local-only or cache semantics explicitly.

## Decision

Transformers.js should treat the value passed as `model` as one of two model sources:

1. A string model ID or local path. Transformers.js calls `OnnxInferenceProvider.from_modelId(modelId)` from `@huggingface/transformers-onnx` and keeps the existing ONNX Runtime behavior.
2. An inference backend object or class. Transformers.js loads shared assets from its `modelId`, calls its `load(options)` method, and never creates an ONNX Runtime session for the model.

The backend boundary is at the model level, not the session level. A custom runtime may have a very different execution model, tensor representation, cache layout, or generation loop, so requiring it to imitate an ORT `InferenceSession` would leak ORT assumptions into the public contract.

The implemented public API is:

```js
import { pipeline } from "@huggingface/transformers";
import { Gemma4E2B } from "@huggingface/webgpu-models";

const pipe = await pipeline("text-generation", Gemma4E2B, {
  dtype: "auto",
});
```

The imported value can be an object or a class with static members. Classes are useful for packages that export one named value per model.

## Backend contract

```ts
interface InferenceBackend {
  /** Hub model ID or local path for config/tokenizer/processor assets. */
  readonly modelId: string;

  /** Load weights, initialize the runtime, and return a model. */
  load(options: InferenceBackendLoadOptions): Promise<InferenceModel>;
}

interface InferenceBackendLoadOptions extends PretrainedModelOptions {
  /** Always supplied by Transformers.js. */
  modelId: string;

  /** Supplied when loading through pipeline(). */
  task?: string;

  /** Resolved PretrainedConfig when loading through pipeline() or AutoModel. */
  config?: PretrainedConfig;
}
```

Transformers.js recognizes the contract structurally: `modelId` must be a string and `load` must be a function. No inheritance, registration, global backend selection, or dependency on an internal base class is required.

An illustrative external model definition is:

```js
export class Gemma4E2B {
  static modelId = "google/gemma-4-e-2b";

  static async load({ dtype, device, progress_callback, config }) {
    const runtime = await WebGPUGemma.load({
      modelId: this.modelId,
      dtype,
      device,
      progress_callback,
    });

    return {
      config,
      forward: (inputs) => runtime.forward(inputs),
      generate: (options) => runtime.generate(options),
      dispose: () => runtime.dispose(),
    };
  }
}
```

`load()` receives a copy of the options. A backend must not rely on mutating the caller's options object.

## Model contract

The object returned by `load()` must implement:

```ts
interface InferenceModel {
  config?: PretrainedConfig;

  forward?(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;

  generate?(
    options: Record<string, unknown>,
  ): Promise<Tensor | GenerationOutput>;

  dispose(): Promise<unknown> | unknown;
}
```

The returned value may instead be directly callable. If it is a plain object with `forward()`, Transformers.js wraps it in a callable proxy so existing pipelines can continue to invoke `model(inputs)`. Other properties and methods, including `generate`, `config`, and `dispose`, are forwarded to the original object.

`dispose()` is required because `Pipeline.dispose()` unconditionally delegates resource cleanup to the model. A backend owns and must release its pipelines, GPU buffers, shader modules, mapped buffers, and device resources.

If a model does not expose `config`, Transformers.js assigns the resolved shared config after `load()`. A backend may supply its own compatible config when necessary.

## Tensor boundary

Pipeline inputs are Transformers.js `Tensor` objects. Model outputs consumed by existing pipelines must also be Transformers.js `Tensor` objects.

This is the remaining shared data-plane contract. `Tensor` is currently backed by an ONNX Runtime tensor internally, so a zero-copy custom WebGPU implementation is not yet possible through every generic tensor operation. The initial custom backend should therefore do one of the following:

1. Convert input tensors to its native representation and return Transformers.js tensors at pipeline-visible boundaries.
2. Own the complete operation, especially generation, and only return the final token IDs or task output tensors.

A later tensor refactor can replace the `ort_tensor` field with a backend-owned native handle. That change is independent of model selection and should preserve the public `Tensor` API.

## Task-specific requirements

The base contract is intentionally small. Each pipeline already has a task-specific model protocol.

### Text generation

The model must implement `generate(options)`. Transformers.js passes tokenizer outputs and user generation options in one object:

```js
const sequences = await model.generate({
  input_ids,
  attention_mask,
  max_new_tokens: 256,
  ...userOptions,
});
```

For decoder-only generation, return an integer `Tensor` shaped `[batch * num_return_sequences, sequence_length]` containing both prompt and generated token IDs. The text-generation pipeline decodes the complete returned sequence.

A custom runtime should normally own its generation loop. Reusing `PreTrainedModel.generate()` currently requires ORT-style session metadata, cache input/output names, and `prepare_inputs_for_generation()` behavior, which is a much larger and less stable interface.

If requested features are supported, `generate()` must honor streamers, stopping criteria, logits processors, sampling options, return dictionaries, and timestamp output. Unsupported options should fail clearly instead of being silently ignored.

### Feature extraction

The model is called with tokenizer output:

```js
const output = await model({ input_ids, attention_mask, ...inputs });
```

It must return at least one of these tensor properties:

```ts
{
    last_hidden_state?: Tensor;
    logits?: Tensor;
    token_embeddings?: Tensor;
}
```

The selected output participates in pooling, slicing, normalization, and quantization in Transformers.js. Mean pooling also uses the tokenizer's `attention_mask`.

### Other pipelines

Existing pipeline classes remain authoritative. Examples:

| Pipeline family               | Required model behavior                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| Classification, QA, detection | Callable model returning the output names expected by that pipeline                         |
| Seq2seq generation            | `generate()` plus compatible `config.prefix` and `config.task_specific_params` when present |
| Image/audio generation        | `generate()` accepting processor tensors under the names used by the pipeline               |
| Image feature extraction      | Callable model returning `pooler_output`, `last_hidden_state`, `logits`, or `image_embeds`  |
| Whisper ASR timestamps        | `generate()` returning `{ sequences, token_timestamps }` when timestamps are requested      |

Supporting a task means implementing that task's existing model protocol; the backend interface does not claim that every backend supports every task.

## Asset resolution

`modelId` separates shared pretrained assets from inference implementation.

Transformers.js uses it for:

- `config.json` through `AutoConfig`
- tokenizer discovery and `AutoTokenizer`
- processor discovery and `AutoProcessor`
- Hub URL construction
- local model paths
- cache keys and file metadata
- revision, cache directory, local-only, and remote/local environment policies
- `ModelRegistry` operations that ultimately resolve model files

The Hub and metadata boundaries normalize backend values to `modelId`, so direct calls can also reuse the descriptor:

```js
const tokenizer = await AutoTokenizer.from_pretrained(AllMiniLML6v2);
const config = await AutoConfig.from_pretrained(AllMiniLML6v2);
const model = await AutoModel.from_pretrained(AllMiniLML6v2);
```

During custom pipeline construction, ONNX model files are excluded from expected-file discovery. Tokenizer and processor files are still auto-detected. The backend is responsible for discovering and downloading its own weight and kernel artifacts.

The first version deliberately requires one shared `modelId`. If weights and tokenizer live in different repositories, the external backend can use its own weight repository internally while setting `modelId` to the repository containing the Transformers-compatible config and tokenizer. Separate `tokenizerId` or `processorId` fields should only be added when a concrete use case requires them.

## Option ownership

Custom `load()` receives the existing pretrained/pipeline options:

| Option                     | Custom backend expectation                                                                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `dtype`                    | Backend-defined weight format. Unsupported values must throw. `Gemma4E2B` initially accepts only omitted/`auto` for its native QAT checkpoint. |
| `device`                   | Select an available target. A backend may support a subset and should reject unsupported values.                                               |
| `progress_callback`        | Report backend-owned weight and initialization progress using existing progress event shapes.                                                  |
| `config`                   | Use the resolved shared config, or return a compatible replacement on the model.                                                               |
| `cache_dir`                | Reuse where the backend's platform and artifact loader support it.                                                                             |
| `local_files_only`         | Do not perform network access when true.                                                                                                       |
| `revision`                 | Resolve backend artifacts from the requested revision.                                                                                         |
| `subfolder`                | May be reused for backend artifacts, but defaults to `onnx` for historical compatibility. A custom backend should not assume it is meaningful. |
| `model_file_name`          | Optional artifact basename override; backend-defined outside ONNX.                                                                             |
| `use_external_data_format` | ONNX-specific and normally ignored by custom runtimes.                                                                                         |
| `session_options`          | ORT-specific today. It is passed through for compatibility but custom backends should not interpret arbitrary ORT settings.                    |
| `task`                     | Pipeline task hint, present only when called through `pipeline()`.                                                                             |

Global `env` policy remains available through the normal Transformers.js export. Custom backends should honor relevant fetch/cache/offline policy rather than introducing conflicting globals.

No generic `backend_options` was added yet. Existing options cover the immediate use case, and adding an untyped escape hatch before two runtimes need the same extension would make the contract less precise.

## Loading sequence

For a custom pipeline:

```text
pipeline(task, backend, options)
  -> validate backend.modelId and backend.load
  -> resolve modelId
  -> discover shared tokenizer/processor files (not ONNX files)
  -> resolve config from modelId
  -> in parallel:
       AutoTokenizer.from_pretrained(modelId, options)
       AutoProcessor.from_pretrained(modelId, options)
       backend.load({ ...options, task, modelId, config })
  -> normalize returned model to the callable model protocol
  -> construct the existing task pipeline
```

For a string:

```text
pipeline(task, modelId, options)
  -> existing AutoModel class selection
  -> PreTrainedModel.from_pretrained(modelId, options)
  -> OnnxInferenceProvider.from_modelId(modelId).load({ ...options, modelClass })
  -> existing session topology
  -> ONNX adapter resolves artifacts and constructs ORT sessions
```

## ONNX Runtime adapter

The ONNX-specific implementation is in the TypeScript package `packages/transformers-onnx`. `packages/transformers/src/models/session.js` remains a small runtime-neutral compatibility facade so existing model implementations do not change.

The adapter owns:

- device-to-ORT execution-provider mapping
- dtype-to-ONNX filename suffix selection
- WebGPU fp16 capability checks
- ONNX model and external-data artifact loading
- ORT session options and free-dimension overrides
- WebGPU preferred output locations for KV caches
- ORT session construction and execution
- WASM proxy input cloning
- Transformers.js tensor to ORT tensor conversion
- ORT output wrapping as Transformers.js tensors
- ORT-specific execution diagnostics

`packages/transformers-onnx/src/runtime.ts` owns Node/web ORT selection, WASM loading, ORT environment defaults, and serialized browser session creation/execution. Core Transformers.js does not import ONNX Runtime packages or expose raw ORT tensors and sessions.

Built-in model forward functions still call `sessionRun()`, and built-in model construction still calls `constructSessions()`. Those compatibility functions now delegate to normalized sessions created by `OnnxInferenceProvider`, preserving existing model implementations and ONNX behavior.

## Why not a session contract?

Current built-in generation reads ORT session details directly:

- `inputNames`
- `inputMetadata`
- symbolic cache shapes
- cache input/output names
- `preferredOutputLocation`
- native tensor locations

Making these public requirements would force a fused WebGPU runtime to expose fake sessions and fake ORT cache metadata. It would also prevent a backend from implementing a faster backend-owned generation loop. A model-level boundary keeps those details private while retaining the high-level pipeline API.

## Errors and validation

Transformers.js rejects malformed backends early:

- no string `modelId`
- no `load(options)` function
- `load()` returns no model
- returned model is neither callable nor has `forward()`
- returned model has no `dispose()`

Task-specific failures, such as a text generation model without `generate()`, surface when the corresponding pipeline invokes that operation. A future task capability declaration could move those errors to pipeline construction, but it is not required for the initial interface.

## Current limitations and follow-ups

1. `Tensor` is still internally coupled to ORT. A backend-neutral native tensor handle is the next major architectural step for zero-copy WebGPU interoperation.
2. Backend-owned weight files are not included in `ModelRegistry.get_pipeline_files()` because Transformers.js cannot infer an external runtime's artifact graph. The backend owns its progress and cache reporting.
3. `session_options`, `subfolder`, and external-data options retain ONNX-oriented names for compatibility. Custom backends should only reuse options with meaningful semantics.
4. Generic generation remains coupled to built-in session metadata. Custom generation backends should implement `generate()`.
5. Pipeline task compatibility is duck-typed. Capability metadata can be added later if early validation becomes valuable.

These limitations do not block the proposed `Gemma4E2B` and `AllMiniLML6v2` API. They keep the initial integration small while establishing a clean ownership boundary between Transformers.js preprocessing/postprocessing and runtime-specific inference.
