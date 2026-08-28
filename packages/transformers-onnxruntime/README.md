# @huggingface/transformers-onnxruntime

ONNX Runtime inference provider for Transformers.js.

```js
import { OnnxInferenceProvider } from '@huggingface/transformers-onnxruntime';

const provider = OnnxInferenceProvider.from_modelId('onnx-community/model-ONNX');
```

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
