# @huggingface/transformers-onnxruntime

ONNX Runtime inference provider for Transformers.js. It maps Transformers.js model classes to ONNX graph files, creates normalized ONNX Runtime sessions, and converts tensors at the runtime boundary.

For the generic provider contract, host services, caching hooks, tensor ownership, and integration requirements, see the [Inference Runtime Integration Guide](../../INFERENCE_RUNTIME_INTEGRATION_GUIDE.md).

## Usage

Transformers.js uses this provider automatically for string model IDs:

```js
import { pipeline } from "@huggingface/transformers";

const classifier = await pipeline(
  "sentiment-analysis",
  "onnx-community/distilbert-base-uncased-finetuned-sst-2-english-ONNX",
  { device: "webgpu", dtype: "q4" },
);
```

Create a provider explicitly when loading a built-in model class directly:

```js
import { AutoModelForSequenceClassification } from "@huggingface/transformers";
import { OnnxInferenceProvider } from "@huggingface/transformers-onnxruntime";

const provider = OnnxInferenceProvider.from_modelId(
  "onnx-community/distilbert-base-uncased-finetuned-sst-2-english-ONNX",
);
const model = await AutoModelForSequenceClassification.from_pretrained(
  provider,
  { device: "webgpu", dtype: "q4" },
);
```

## Provider API

`OnnxInferenceProvider` is a normalized-session provider. Transformers.js owns model semantics and calls the provider to resolve and construct the ONNX sessions required by each architecture.

- `OnnxInferenceProvider.from_modelId(modelId)` creates a provider for one model repository.
- `constructSessions(names, options, cacheSessions?)` loads graphs and creates normalized sessions.
- `getSessionConfig(modelType, config, options?)` resolves architecture-specific session roles and files.
- `OnnxInferenceProvider.listModelArtifacts(options)` returns required ONNX graphs, external-data chunks, and adjacent config files.
- `OnnxInferenceProvider.getAvailableDtypes(options)` checks which complete dtype variants exist.
- `OnnxInferenceProvider.filterModelArtifacts(files, options)` removes graph files not used by the selected model path.

The provider supports per-session `device` and `dtype` maps in addition to scalar values. It resolves standard ONNX filename suffixes, external-data chunks, cache-session settings, and text-only subsets for supported multimodal architectures.

## Runtime

Install Transformers.js for the default browser or Node path:

```sh
npm install @huggingface/transformers
```

Applications importing this provider directly in Node must also install the optional Node runtime peer:

```sh
npm install @huggingface/transformers-onnxruntime onnxruntime-node
```

Browser builds use `onnxruntime-web`; Node builds use `onnxruntime-node`. WASM assets load from the pinned `onnxruntime-web` CDN by default. Set `env.backends.onnx.wasm.wasmPaths` before loading a model to serve them from another location.
