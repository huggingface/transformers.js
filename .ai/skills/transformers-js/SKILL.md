---
name: transformers-js
description: Run state-of-the-art machine learning models directly in JavaScript. `@huggingface/transformers` supports text, vision, audio, and multimodal tasks in browsers and Node.js / Bun / Deno, with WebGPU or WASM execution.
license: Apache-2.0
metadata:
  author: huggingface
  repository: https://github.com/huggingface/transformers.js
compatibility: Node.js 18+ (or equivalent Bun / Deno), or a modern browser with ES modules. WebGPU requires runtime and hardware support; WASM is the fallback. Model downloads from the Hugging Face Hub require network access unless you ship models locally.
---

# transformers.js

ML inference for JavaScript, without a Python server. Supports text, vision, audio,
and multimodal tasks through a single `pipeline()` entry point.

## Install

```bash
npm install @huggingface/transformers
```

## Quick start

```javascript
import { pipeline } from "@huggingface/transformers";

const classifier = await pipeline("sentiment-analysis");
const output = await classifier("I love transformers!");
// [{ label: "POSITIVE", score: 0.9998 }]
```

`pipeline(task, model?, options?)` is the one function you need 90% of the time.
Passing no `model` uses the default for that task.

## Supported tasks

<!-- @generated:start id=task-list -->
- [`text-classification`](references/TASKS.md#text-classification) _(alias: `sentiment-analysis`)_ — default model: `Xenova/distilbert-base-uncased-finetuned-sst-2-english`
- [`token-classification`](references/TASKS.md#token-classification) _(alias: `ner`)_ — default model: `Xenova/bert-base-multilingual-cased-ner-hrl`
- [`question-answering`](references/TASKS.md#question-answering) — default model: `Xenova/distilbert-base-cased-distilled-squad`
- [`fill-mask`](references/TASKS.md#fill-mask) — default model: `onnx-community/ettin-encoder-32m-ONNX`
- [`summarization`](references/TASKS.md#summarization) — default model: `Xenova/distilbart-cnn-6-6`
- [`translation`](references/TASKS.md#translation) — default model: `Xenova/t5-small`
- [`text2text-generation`](references/TASKS.md#text2text-generation) — default model: `Xenova/flan-t5-small`
- [`text-generation`](references/TASKS.md#text-generation) — default model: `onnx-community/Qwen3-0.6B-ONNX`
- [`zero-shot-classification`](references/TASKS.md#zero-shot-classification) — default model: `Xenova/distilbert-base-uncased-mnli`
- [`audio-classification`](references/TASKS.md#audio-classification) — default model: `Xenova/wav2vec2-base-superb-ks`
- [`zero-shot-audio-classification`](references/TASKS.md#zero-shot-audio-classification) — default model: `Xenova/clap-htsat-unfused`
- [`automatic-speech-recognition`](references/TASKS.md#automatic-speech-recognition) _(alias: `asr`)_ — default model: `Xenova/whisper-tiny.en`
- [`text-to-audio`](references/TASKS.md#text-to-audio) _(alias: `text-to-speech`)_ — default model: `onnx-community/Supertonic-TTS-ONNX`
- [`image-to-text`](references/TASKS.md#image-to-text) — default model: `Xenova/vit-gpt2-image-captioning`
- [`image-classification`](references/TASKS.md#image-classification) — default model: `Xenova/vit-base-patch16-224`
- [`image-segmentation`](references/TASKS.md#image-segmentation) — default model: `Xenova/detr-resnet-50-panoptic`
- [`background-removal`](references/TASKS.md#background-removal) — default model: `Xenova/modnet`
- [`zero-shot-image-classification`](references/TASKS.md#zero-shot-image-classification) — default model: `Xenova/clip-vit-base-patch32`
- [`object-detection`](references/TASKS.md#object-detection) — default model: `Xenova/detr-resnet-50`
- [`zero-shot-object-detection`](references/TASKS.md#zero-shot-object-detection) — default model: `Xenova/owlvit-base-patch32`
- [`document-question-answering`](references/TASKS.md#document-question-answering) — default model: `Xenova/donut-base-finetuned-docvqa`
- [`image-to-image`](references/TASKS.md#image-to-image) — default model: `Xenova/swin2SR-classical-sr-x2-64`
- [`depth-estimation`](references/TASKS.md#depth-estimation) — default model: `onnx-community/depth-anything-v2-small`
- [`feature-extraction`](references/TASKS.md#feature-extraction) _(alias: `embeddings`)_ — default model: `onnx-community/all-MiniLM-L6-v2-ONNX`
- [`image-feature-extraction`](references/TASKS.md#image-feature-extraction) — default model: `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`
<!-- @generated:end id=task-list -->

For full recipes — every task, grouped by modality, with runnable code —
see [`references/TASKS.md`](references/TASKS.md).

## Choosing a model

Browse models compatible with transformers.js on the Hub:
<https://huggingface.co/models?library=transformers.js>

Filter by task with the `pipeline_tag` parameter, e.g.
<https://huggingface.co/models?library=transformers.js&pipeline_tag=text-generation>.

**Before recommending a model**, confirm it actually has ONNX weights — the
library cannot load a model without them. Two ways to check:

1. Open the model page on the Hub and look for an `onnx/` directory in the
   "Files and versions" tab.
2. Programmatically, with `ModelRegistry.get_available_dtypes(modelId)` —
   returns the list of dtypes shipped. An empty array means the model exists
   but ships no ONNX files. It throws a `ModelFileNotFoundError` if the model
   does not exist or is not accessible (private/gated — the Hub cannot tell
   these apart), and a regular error on network failures, so an unreachable
   model is never mistaken for one without ONNX files.

```javascript
import { ModelRegistry, ModelFileNotFoundError } from "@huggingface/transformers";

// "Xenova/some-model" is a placeholder — substitute the ID you want to check.
try {
  const dtypes = await ModelRegistry.get_available_dtypes("Xenova/some-model");
  if (dtypes.length === 0) {
    // Model exists but has no ONNX files — not usable with transformers.js.
  }
} catch (e) {
  if (e instanceof ModelFileNotFoundError) {
    // Model does not exist, or is private/gated without a token.
  } else {
    throw e; // network failure — retry rather than blacklisting the model
  }
}
```

Don't suggest a model without verifying this; the failure mode at runtime is a
download error that's harder to diagnose than a pre-flight check. For a fuller
pre-flight pattern (cache checks, dtype fallback), see
[`references/CONFIGURATION.md`](references/CONFIGURATION.md#inspecting-models-before-loading).

### Quantization

Most pipelines accept a `dtype` option. Smaller dtypes download and run faster
at the cost of some accuracy:

| `dtype`  | Size      | Use when                                           |
|----------|-----------|----------------------------------------------------|
| `fp32`   | Largest   | Maximum accuracy, Node.js with lots of RAM         |
| `fp16`   | ~50% of fp32 | GPU / WebGPU inference                          |
| `q8`     | ~25% of fp32 | Good default for browsers                       |
| `q4`     | ~12% of fp32 | Tight memory budgets, large language models     |
| `q4f16`  | ~12% of fp32 | Like `q4` but with fp16 activations — pairs well with WebGPU LLMs |

```javascript
const pipe = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX", {
  dtype: "q4",
});
```

### Device

Default is CPU/WASM. Pass `device: "webgpu"` to run on the GPU when available:

```javascript
const pipe = await pipeline("sentiment-analysis", null, { device: "webgpu" });
```

## Memory management

Pipelines hold onto model weights and backend sessions. **Always call
`pipe.dispose()`** when you're done with one — especially in long-running
servers, before loading a replacement, or on component unmount.

```javascript
const pipe = await pipeline("sentiment-analysis");
try {
  const result = await pipe("Great!");
} finally {
  await pipe.dispose();
}
```

## Configuration

The [`env`](https://huggingface.co/docs/transformers.js/api/env) export lets
you control model sources, caching, logging, and the fetch function.

```javascript
import { env, LogLevel } from "@huggingface/transformers";

env.allowRemoteModels = true;
env.useFSCache = true;           // Node.js: cache downloaded models on disk
env.useBrowserCache = true;      // Browser: cache via the Cache API
env.logLevel = LogLevel.WARNING;
```

See [`references/CONFIGURATION.md`](references/CONFIGURATION.md) for the full
set of environment options, cache management, and private / gated models.

## Pipeline options

Every pipeline accepts a `progress_callback` for download progress plus options
controlling device, dtype, and caching. The per-task recipes in
[`references/TASKS.md`](references/TASKS.md) show common call options (e.g.
`top_k`, `max_new_tokens`) in use; shared loading options, generation
parameters, streaming, and KV-cache reuse are documented in
[`references/PIPELINE_OPTIONS.md`](references/PIPELINE_OPTIONS.md). For the
exhaustive per-task option types, see the
[API reference](https://huggingface.co/docs/transformers.js/api/pipelines).

## Things to never do

- **Don't reuse a disposed pipeline.** Create a new one with `pipeline(...)` after `dispose()`.
- **Don't recreate pipelines inside hot loops.** Create once, call many times.
- **Don't block startup on model downloads.** Show progress via `progress_callback`.
- **Don't fabricate model IDs.** Confirm a model exists on the Hub and has ONNX files
  (look for an `onnx/` directory in the repo) before suggesting it to a user.

## Reference documentation

- Official site: <https://huggingface.co/docs/transformers.js>
- API reference: <https://huggingface.co/docs/transformers.js/api/pipelines>
- Examples repo: <https://github.com/huggingface/transformers.js-examples>

This skill's local references:

- [`TASKS.md`](references/TASKS.md) — recipes for every task, grouped by modality _(generated)_
- [`CONFIGURATION.md`](references/CONFIGURATION.md) — `env` options, caching, model inspection
- [`PIPELINE_OPTIONS.md`](references/PIPELINE_OPTIONS.md) — common pipeline options, dtype, device, generation parameters
