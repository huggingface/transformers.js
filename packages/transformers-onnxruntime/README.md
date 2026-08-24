# @huggingface/transformers-onnxruntime

ONNX Runtime inference provider for Transformers.js.

```js
import { OnnxInferenceProvider } from '@huggingface/transformers-onnxruntime';

const provider = OnnxInferenceProvider.from_modelId('onnx-community/model-ONNX');
```

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
