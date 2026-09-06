# Running models on WebGPU

WebGPU is a new web standard for accelerated graphics and compute. The [API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) enables web developers to use the underlying system's GPU to carry out high-performance computations directly in the browser. WebGPU is the successor to [WebGL](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API) and provides significantly better performance, because it allows for more direct interaction with modern GPUs. Lastly, it supports general-purpose GPU computations, which makes it just perfect for machine learning!

> [!WARNING]  
> As of October 2024, global WebGPU support is around 70% (according to [caniuse.com](https://caniuse.com/webgpu)), meaning some users may not be able to use the API.
>
> If the following demos do not work in your browser, you may need to enable it using a feature flag:
>
> - Firefox: with the `dom.webgpu.enabled` flag (see [here](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features#:~:text=tested%20by%20Firefox.-,WebGPU%20API,-The%20WebGPU%20API)).
> - Safari: with the `WebGPU` feature flag (see [here](https://webkit.org/blog/14879/webgpu-now-available-for-testing-in-safari-technology-preview/)).
> - Older Chromium browsers (on Windows, macOS, Linux): with the `enable-unsafe-webgpu` flag (see [here](https://developer.chrome.com/docs/web-platform/webgpu/troubleshooting-tips)).

## Usage in Transformers.js v3

Thanks to our collaboration with [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), enabling WebGPU acceleration is as simple as setting `device: 'webgpu'` when loading a model. Let's see some examples!

**Example:** Compute text embeddings on WebGPU ([demo](https://v2.scrimba.com/s06a2smeej))

```js
import { pipeline } from "@huggingface/transformers";

// Create a feature-extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "mixedbread-ai/mxbai-embed-xsmall-v1",
  { device: "webgpu" },
);

// Compute embeddings
const texts = ["Hello world!", "This is an example sentence."];
const embeddings = await extractor(texts, { pooling: "mean", normalize: true });
console.log(embeddings.tolist());
// [
//   [-0.016986183822155, 0.03228696808218956, -0.0013630966423079371, ... ],
//   [0.09050482511520386, 0.07207386940717697, 0.05762749910354614, ... ],
// ]
```

**Example:** Perform automatic speech recognition with OpenAI whisper on WebGPU ([demo](https://v2.scrimba.com/s0oi76h82g))

```js
import { pipeline } from "@huggingface/transformers";

// Create automatic speech recognition pipeline
const transcriber = await pipeline(
  "automatic-speech-recognition",
  "onnx-community/whisper-tiny.en",
  { device: "webgpu" },
);

// Transcribe audio from a URL
const url =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
const output = await transcriber(url);
console.log(output);
// { text: ' And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.' }
```

**Example:** Perform image classification with MobileNetV4 on WebGPU ([demo](https://v2.scrimba.com/s0fv2uab1t))

```js
import { pipeline } from "@huggingface/transformers";

// Create image classification pipeline
const classifier = await pipeline(
  "image-classification",
  "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
  { device: "webgpu" },
);

// Classify an image from a URL
const url =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg";
const output = await classifier(url);
console.log(output);
// [
//   { label: 'tiger, Panthera tigris', score: 0.6149784922599792 },
//   { label: 'tiger cat', score: 0.30281734466552734 },
//   { label: 'tabby, tabby cat', score: 0.0019135422771796584 },
//   { label: 'lynx, catamount', score: 0.0012161266058683395 },
//   { label: 'Egyptian cat', score: 0.0011465961579233408 }
// ]
```

## Decode graph capture

For compatible decoder models, set `session_options.enableGraphCapture: true`
to capture and replay single-token decode steps. Prefill always skips capture,
including one-token prompts and multi-token continuations of an existing cache.
Encoders and other component sessions also run normally.

This requires an ONNX export built for static KV caches and WebGPU graph capture,
such as an ORT GenAI export with `enable_webgpu_graph=true`. Ordinary exports that
concatenate an increasingly long KV cache are not compatible. This option does
not convert the ONNX graph. The implementation uses ONNX Runtime Web's C++ WebGPU
EP, which is the runtime imported by Transformers.js; the older JSEP backend and
the native Node.js runtime are not supported.

For a compatible export packaged for Transformers.js:

```js
import { AutoConfig, AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";

const modelId = "./phi4-webgpu";
const config = await AutoConfig.from_pretrained(modelId);
config["transformers.js_config"] = {
  ...config["transformers.js_config"],
  max_cache_length: 2048, // Capacity includes the prompt and generated tokens.
};
const model = await AutoModelForCausalLM.from_pretrained(modelId, {
  config,
  device: "webgpu",
  dtype: "q4f16",
  session_options: { enableGraphCapture: true },
});
const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const inputs = tokenizer("<|user|>What is the capital of France?<|end|><|assistant|>");
const output = await model.generate({ ...inputs, max_new_tokens: 128, do_sample: false });
console.log(tokenizer.decode(output[0].tolist(), { skip_special_tokens: true }));
await model.dispose();
```

The same loading options work with `pipeline("text-generation", ...)`.
Batch size is currently limited to one. Do not override the prompt dimensions
with `freeDimensionOverrides`: prefill must accept the actual prompt length.

Transformers.js allocates fixed-capacity KV buffers and binds each cache output
to its corresponding input buffer. Decode uses stable GPU token, mask, and logit
buffers. Only logits are downloaded for sampling. `max_cache_length` defaults to
2048 and exceeding it raises an error before running the model.

A static cache's tensor dimensions describe its capacity;
`past_key_values.get_seq_length()` reports the number of processed tokens.
The buffers belong to the model and are released by `model.dispose()`. Returned
KV tensors are views of that mutable cache. Start only one generation at a time
per model instance. A new independent prompt invalidates caches retained from
an older sequence; pass the current `past_key_values` to continue that sequence.

For a controlled capture-off baseline, set
`config["transformers.js_config"].use_static_cache = true` and
`session_options.enableGraphCapture = false`. This keeps the export, cache
capacity, and shared buffers the same while disabling capture.

To benchmark a local graph-ready ORT GenAI Phi-4-mini export through the actual
Transformers.js generation path, use the repository's benchmark script:

```sh
node packages/transformers/scripts/benchmarks/graph-capture.mjs \
  --model /path/to/Phi-4-mini-instruct \
  --webgpu-module /path/to/dawn.node \
  --prompt-tokens 128 --new-tokens 128 --context 2048 \
  --warmup 2 --runs 5 --output graph-capture-results.json
```

Use Node.js 24 or newer and a Dawn binding with `shader-f16` and `subgroups`
(on Windows, build Dawn with DXC). The script uses ORT Web's Asyncify build,
compares identical generated token IDs, and reports decode throughput separately
from time to first token. Model loading and warmup are excluded from decode
measurements. The initial decode calls prepare and capture the graph, so compare
warmed-up runs; performance depends on the model, context capacity, and hardware.

See the [ONNX Runtime WebGPU documentation](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html#graph-capture)
for the runtime requirements.
## Reporting bugs and providing feedback

Due to the experimental nature of WebGPU, especially in non-Chromium browsers, you may experience issues when trying to run a model (even if it can run in WASM). If you do, please open [an issue on GitHub](https://github.com/huggingface/transformers.js/issues/new?title=[WebGPU]%20Error%20running%20MODEL_GOES_HERE&assignees=&labels=bug,webgpu&projects=&template=1_bug-report.yml) and we'll do our best to address it. Thanks!
