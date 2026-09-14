# @huggingface/transformers-onnxruntime

ONNX Runtime inference provider for Transformers.js.

```js
import { OnnxInferenceProvider } from '@huggingface/transformers-onnxruntime';

const provider = OnnxInferenceProvider.from_modelId('onnx-community/model-ONNX');
```

## InferenceBackend

Transformers.js defines `InferenceBackend` as a structural interface in
`packages/transformers/src/backends/inference.js`. At minimum, a backend has a shared model ID
and a loader:

```ts
type InferenceBackend = {
    modelId: string;
    load(options: InferenceBackendLoadOptions): Promise<InferenceModel | Function>;

    sharedAssets?: { revision?: string; subfolder?: string };
    chatTemplate?:
        | { content: string }
        | { modelId?: string; revision?: string; subfolder?: string; file?: string };
    capabilities?: { devices: string[]; dtypes: string[]; tasks: string[] };

    listModelArtifacts?(options: object): string[] | Promise<string[]>;
    getModelArtifactMetadata?(file: string, options: object): Promise<object | null>;
    deleteModelArtifact?(file: string, options: object): Promise<boolean>;
};
```

`modelId` identifies the config, tokenizer, processor, and other shared Transformers.js assets.
`sharedAssets`, `chatTemplate`, and `capabilities` are optional backend metadata, while the
artifact methods let a backend participate in file discovery, progress reporting, and cache
management. `load()` receives the resolved model ID, the host's `fetch`, normal pretrained-model
options, and, when available, the parsed config, selected task and model class, generation config,
and artifact metadata.

A runtime-neutral backend returns either a callable model or an object implementing
`forward(inputs)` and/or a task-specific entry point such as `createAutoregressiveSession()` for
causal generation. Every returned model must implement `dispose()` and can declare its execution
capabilities. Transformers.js normalizes objects with `forward()` into callable models and installs
the public generation runtime around the autoregressive session contract.

`OnnxInferenceProvider` implements the `modelId` and `load()` entry shape, but it is a specialized
backend for built-in Transformers.js model classes. Transformers.js recognizes it through
`providerType === 'onnx'` and `constructSessions()`. Its `load()` receives the semantic model class
selected by Transformers.js and calls that class's `_from_pretrained()` with itself as the
`inferenceProvider`; the model class then asks the provider for the architecture's session mapping
and construction. `constructSessions()` creates normalized ONNX Runtime sessions, and `run()`
converts between Transformers.js tensors and ONNX Runtime tensors. Thus Transformers.js continues
to own model semantics and public model behavior, while this package owns ONNX artifacts, runtime
sessions, execution providers, dtypes, and tensor interop.

For ordinary string model IDs, Transformers.js creates an `OnnxInferenceProvider` automatically.
The provider's static artifact-discovery methods are also used by the model registry to enumerate
ONNX files, discover available dtypes, and filter architecture-specific artifacts.

## Transformers.js boundary

Transformers.js owns semantic model behavior: it parses `config.json`, selects the public
model class, and classifies the architecture as encoder-only, decoder-only, seq2seq,
multimodal, and so on. This package owns the ONNX representation of that category.

Given the parsed config and semantic category, `OnnxInferenceProvider` resolves:

- the required ONNX sessions and filenames;
- optional ONNX-adjacent files such as `generation_config.json`;
- dtype suffixes and external-data chunks;
- cache-session flags and text-only multimodal subsets.

The same provider-owned mapping is used by model registry discovery, dtype discovery, and
actual session construction. Transformers.js does not precompute a `sessions` map for the
provider. String model IDs are converted to `OnnxInferenceProvider` instances by default, so
`pipeline(task, modelId)` and `AutoModel.from_pretrained(modelId)` use this boundary without
additional application configuration.

Transformers.js configures the provider host before loading. The host supplies model-file
transport, cache access, tensors, logging, environment capabilities, and the current
`env.fetch`; the provider does not independently fall back to `globalThis.fetch`.

## Runtime dependencies

`@huggingface/transformers` installs the Node runtime required by its default ONNX model path:

```sh
npm install @huggingface/transformers
```

Applications installing this provider directly must also install its optional Node peer when they use ONNX models in Node:

```sh
npm install @huggingface/transformers-onnxruntime onnxruntime-node
```

Browser ESM builds load this package lazily through the bare `@huggingface/transformers-onnxruntime` specifier. Direct CDN usage therefore requires an import map for this package, `onnxruntime-common`, and `onnxruntime-web/webgpu`. The main Transformers.js installation guide contains a complete example. By default, WASM assets load from the pinned `onnxruntime-web` CDN; set `env.backends.onnx.wasm.wasmPaths` before loading a model to host them elsewhere.
