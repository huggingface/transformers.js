# @huggingface/transformers-litertjs-runtime

> [!WARNING] This entire package is highly experimental. Its APIs, model descriptors, artifact locations, preprocessing choices, runtime compatibility, and numerical behavior may change without notice. Validate outputs for your application before relying on them in production.

LiteRT.js inference provider for Transformers.js. It supports selected fixed-shape models highlighted in Google's [LiteRT.js launch article](https://developers.googleblog.com/litertjs-googles-high-performance-web-ai-inference/):

| Provider model ID | Pipeline task | Default artifact |
| --- | --- | --- |
| `litert-community/efficientnet_b0` | `image-classification` | EfficientNet-B0 FP32 |
| `litert-community/inception_v3` | `image-classification` | Inception V3 FP32 |
| `litert-community/resnet50` | `image-classification` | ResNet-50 FP32 |
| `litert-community/whisper-tiny` | `automatic-speech-recognition` | Whisper Tiny int8 |
| `litert-community/whisper-base` | `automatic-speech-recognition` | Whisper Base int8 |
| `litert-community/yolo26n` | `object-detection` | YOLO26n W8A32 v0.6.6 |

Google did not publish the exact benchmark files, revisions, hashes, or harness. The descriptors use pinned public candidate artifacts and must not be interpreted as the exact benchmark inputs. The YOLO26n descriptor targets an official post-launch Ultralytics artifact because no launch-time artifact was published.

## Usage

```js
import { pipeline } from '@huggingface/transformers';
import { LiteRtInferenceProvider } from '@huggingface/transformers-litertjs-runtime';

const provider = LiteRtInferenceProvider.from_modelId(
    'litert-community/efficientnet_b0',
);
const classifier = await pipeline('image-classification', provider, {
    dtype: 'fp32',
    device: 'webgpu',
});

const result = await classifier(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png',
);
```

Whisper Tiny and Base use the model's `encode` and `decode` signatures for greedy transcription while Transformers.js owns feature extraction, tokenization, and ASR postprocessing:

```js
const provider = LiteRtInferenceProvider.from_modelId(
    'litert-community/whisper-base',
);
const transcriber = await pipeline('automatic-speech-recognition', provider, {
    dtype: 'q8',
    device: 'wasm',
});
```

YOLO26n uses model-owned 640x640 letterbox preprocessing, inverse coordinate mapping, confidence filtering, and class-aware NMS because its raw `[1, 84, 8400]` output is not compatible with DETR postprocessing. GitHub release assets may not be directly fetchable in every browser due to CORS. Mirror the pinned file on the same origin and override its source when necessary:

```js
const provider = LiteRtInferenceProvider.from_modelId(
    'litert-community/yolo26n',
    {
        artifactModelId: '/models',
        modelFile: 'yolo26n_w8a32.tflite',
    },
);

const detector = await pipeline('object-detection', provider, {
    dtype: 'q8',
    device: 'webgpu',
});
```

The three classifiers currently reuse compatible Transformers.js ImageNet shared assets. Inception input is resized and transposed to its `[1, 299, 299, 3]` runtime layout inside the provider. These shared processor choices are experimental and are not guaranteed to reproduce the source framework's exact validation recipe.

## Runtime and caching

`@litertjs/core` is a browser runtime. Its WASM files are loaded from jsDelivr by default. Set `wasmPath` to a directory containing the package's `wasm/` assets when serving them yourself.

TFLite artifacts are loaded through the cache-aware `getModelFile()` service supplied by Transformers.js. They use the configured transport, progress reporting, deduplication, `local_files_only` behavior, and browser cache. Model registry metadata and cache deletion delegate to the same host services.

Whisper supports greedy generation without timestamps. `q8` and `int8` select the compressed artifact; `fp32` selects the float artifact where available. EfficientNet-B0 and ResNet-50 also expose weight-compressed `q8` variants with float32 model I/O. Inception V3 currently supports only `fp32`, and the pinned YOLO26n descriptor currently supports only `q8`/`int8`.
