# @huggingface/transformers-onnxruntime

ONNX Runtime inference provider for Transformers.js.

```js
import { OnnxInferenceProvider } from '@huggingface/transformers-onnxruntime';

const provider = OnnxInferenceProvider.from_modelId('onnx-community/model-ONNX');
```

## Runtime dependencies

Node applications using the ONNX provider must install `onnxruntime-node` alongside Transformers.js:

```sh
npm install @huggingface/transformers onnxruntime-node
```

The Node runtime is an optional peer so browser-only and custom-backend-only installations do not download native ONNX binaries.

Browser ESM builds load this package lazily through the bare `@huggingface/transformers-onnxruntime` specifier. Direct CDN usage therefore requires an import map for this package and its `onnxruntime-web` dependencies. Keep the provider's copied `.mjs` and `.wasm` files beside `transformers-onnxruntime.web.js`; default relative `wasmPaths` are resolved from that module URL. Alternatively, set `env.backends.onnx.wasm.wasmPaths` before loading a model.
