# LiteRT.js benchmark model semantics

## Scope and conclusion

This note defines the task-level semantics needed to expose the likely LiteRT.js launch-benchmark models through Transformers.js. It separates facts verified from public artifacts and source code from implementation recommendations. Exact launch artifacts remain undisclosed, so the inspected LiteRT Community files are reproducible candidates, not proven benchmark inputs.

The three classifiers can use the existing `image-classification` pipeline after exact preprocessing and runtime layout adaptation. Whisper Base can use the existing `automatic-speech-recognition` pipeline only through a Whisper-specific generation adapter. Ultralytics YOLO cannot use the current DETR/YOLOS postprocessor: it needs letterbox-aware preprocessing, raw-head decoding, confidence filtering, and NMS.

| Candidate | Transformers.js task | Pipeline-ready? | Required adapter |
|---|---|---:|---|
| `litert-community/efficientnet_b0` | `image-classification` | Yes, after sidecars are supplied | Map the single output to `logits` |
| `litert-community/inception_v3` | `image-classification` | Yes, after sidecars are supplied | Transpose processor NCHW to model NHWC; map output to `logits` |
| `litert-community/resnet50` | `image-classification` | Yes, after sidecars are supplied | Map the single output to `logits` |
| `litert-community/whisper-base` | `automatic-speech-recognition` | Not through generic forward | Whisper feature extraction, iterative decoding, generation rules, and tokenizer decoding |
| YOLO11n or YOLO26n | `object-detection` | No, not with the current postprocessor | Ultralytics letterbox and raw-output postprocessor, including NMS |

## Benchmark naming caveat

Google's article contains two different detector labels. The classical-model accelerator table says `Yolo11n`; the normalized five-model comparison chart says `YOLO26n`. Both labels should be retained in research and test names rather than silently normalizing one to the other. The article does not identify the detector filename, hash, export version, input resolution, output-head mode, or quantization.

The inspected `yolo26n_w8a32.tflite` is an official post-launch Ultralytics release artifact. It is useful for defining one concrete YOLO26 integration contract, but it is not evidence of the exact file Google benchmarked. See [the launch article](https://developers.googleblog.com/litertjs-googles-high-performance-web-ai-inference/), [the normalized chart](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/Data_image_1600x900.original.png), and [the accelerator table](https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/Classical_model_perf_1.original.png).

## Image classifiers

### Shared output contract

All three inspected classifier candidates produce one FP32 tensor with shape `[1, 1000]`. These are pre-softmax ImageNet-1K logits. The provider should return that tensor as `{ logits }` without applying softmax. `ImageClassificationPipeline` already applies softmax, computes top-k, and resolves class indices through `model.config.id2label`; applying softmax in the provider would be redundant and could change scores. See [`image-classification.js`](../../packages/transformers/src/pipelines/image-classification.js) and the [EfficientNet](https://huggingface.co/litert-community/efficientnet_b0), [Inception](https://huggingface.co/litert-community/inception_v3), and [ResNet](https://huggingface.co/litert-community/resnet50) cards.

The cards use the standard ImageNet-1K class order from [`huggingface/label-files/imagenet-1k-id2label.json`](https://huggingface.co/datasets/huggingface/label-files/blob/main/imagenet-1k-id2label.json). A compatible `config.json` must include that exact `id2label` mapping. Falling back to `LABEL_n` would run correctly but would not provide the intended user-facing semantics.

### EfficientNet-B0

Verified artifact contract:

- Signature: `main`.
- Input: logical `x`, runtime tensor `g0::x`, FP32 `[1, 3, 224, 224]`, NCHW.
- Output: logical `linear`, runtime tensor `g0::linear`, FP32 `[1, 1000]` logits.
- RGB preprocessing: resize the shorter edge to 256 with bicubic interpolation, center-crop to 224 by 224, divide byte values by 255, then normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

The transform above is stated directly in the candidate repository's usage code and matches the TorchVision EfficientNet-B0 V1 recipe. The model card's reported source accuracy also matches the V1 checkpoint. See the [LiteRT model card](https://huggingface.co/litert-community/efficientnet_b0) and [TorchVision weights documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html).

Do not reuse `google/efficientnet-b0/preprocessor_config.json`. It describes a different resize path and uses `EfficientNetImageProcessor`; the local implementation squares each configured standard deviation when `include_top` is true. That is correct for the Google Transformers checkpoint's convention, but not for this TorchVision export. See the [Google processor config](https://huggingface.co/google/efficientnet-b0/blob/main/preprocessor_config.json) and [`image_processing_efficientnet.js`](../../packages/transformers/src/models/efficientnet/image_processing_efficientnet.js).

Recommended sidecar semantics use the base `ImageProcessor`: `size.shortest_edge = 256`, `crop_size = {width: 224, height: 224}`, `do_center_crop = true`, bicubic resampling, `do_rescale = true`, `rescale_factor = 1/255`, and the ImageNet mean/std above. The processor already emits NCHW, so no provider layout transform is needed.

### Inception-v3

Verified artifact contract:

- Signature: `serving_default`.
- Input: logical `args_0`, runtime tensor `serving_default_args_0:0`, FP32 `[1, 299, 299, 3]`, NHWC.
- Output: logical `output_0`, runtime tensor `StatefulPartitionedCall:0`, FP32 `[1, 1000]` logits.
- RGB preprocessing: resize the shorter edge to 342 with bilinear interpolation, center-crop to 299 by 299, divide by 255, then normalize with ImageNet mean/std.

The candidate card explicitly identifies `Inception_V3_Weights.IMAGENET1K_V1` and provides the transform above. See the [LiteRT model card](https://huggingface.co/litert-community/inception_v3) and [TorchVision weights documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html).

The base Transformers.js image processor produces planar NCHW `pixel_values`. The provider must transpose `[1, 3, 299, 299]` to `[1, 299, 299, 3]` immediately before constructing the LiteRT input. This is a model descriptor property, not a different pipeline contract.

Recommended sidecar semantics use `size.shortest_edge = 342`, a 299-by-299 center crop, bilinear resampling, `1/255` rescaling, and ImageNet mean/std. No existing Transformers.js Inception repository was found that both carries the intended TorchVision V1 labels and expresses this exact transform, so model-local sidecars are safer than borrowing a TIMM checkpoint's metadata.

### ResNet-50

Verified artifact contract:

- Signature: `main`.
- Input: logical `x`, runtime tensor `g0::x`, FP32 `[1, 3, 224, 224]`, NCHW.
- Output: logical `linear`, runtime tensor `g0::linear`, FP32 `[1, 1000]` logits.
- RGB preprocessing: resize the shorter edge to 232 with bilinear interpolation, center-crop to 224 by 224, divide by 255, then normalize with ImageNet mean/std.

The candidate card reports 80.858% top-1 and 95.434% top-5 source accuracy, identifying the TorchVision V2/default weights rather than V1. The V2 inference preset uses the 232-pixel resize. See the [LiteRT model card](https://huggingface.co/litert-community/resnet50) and [TorchVision ResNet-50 weights documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).

`microsoft/resnet-50` provides a compatible standard ImageNet label map, but its current processor config uses a 224 target with `crop_pct = 0.875`; the local `ConvNextFeatureExtractor` implementation consequently resizes the shorter edge to 256, not 232. It is therefore not an exact preprocessing sidecar for this LiteRT artifact. See the [processor config](https://huggingface.co/microsoft/resnet-50/blob/main/preprocessor_config.json) and [`image_processing_convnext.js`](../../packages/transformers/src/models/convnext/image_processing_convnext.js).

Recommended sidecar semantics use the base `ImageProcessor`: `size.shortest_edge = 232`, a 224-by-224 center crop, bilinear resampling, `1/255` rescaling, and ImageNet mean/std. It emits the required NCHW layout.

### Classifier sidecar recommendation

Publish exact `config.json` and `preprocessor_config.json` files with each LiteRT repository, or publish dedicated shared-assets repositories whose revisions are pinned by tests. Reusing a nearby architecture repository solely for labels also reuses its processor configuration, which can silently alter resize, crop, interpolation, or normalization.

At minimum, each classifier config needs 1,000 ImageNet labels. Each processor config should select the base `ImageProcessor`, convert to RGB, and encode the model-specific transform above. The provider descriptor should separately encode runtime layout (`nchw` or `nhwc`) and semantic output name (`logits`); converter-generated tensor names should only be validated against the selected pinned artifact.

## Whisper Base

### Verified tensor protocol

Both inspected Whisper Base variants expose the same external signatures:

| Signature | Inputs | Output |
|---|---|---|
| `encode` | FP32 log-Mel features `[1, 80, 3000]` | FP32 encoder state `[1, 1500, 512]` |
| `decode` | FP32 encoder state `[1, 1500, 512]`; INT32 token IDs `[1, 128]`; FP32 mask `[1, 1, 128, 128]` | FP32 vocabulary logits `[1, 128, 51865]` |

The files are fixed to one 30-second audio window and a maximum decoder buffer of 128 positions. The public card verifies the shapes but does not document the converter source or the numeric convention for the explicit decoder mask. See the [Whisper Base LiteRT card](https://huggingface.co/litert-community/whisper-base).

### Reusable upstream assets

`openai/whisper-base` is the correct shared-assets source for:

- `config.json`, including `d_model = 512`, vocabulary size 51,865, token IDs, and generation-related architecture values.
- `generation_config.json`, including language/task prompt mappings and suppression settings.
- Tokenizer vocabulary, merges, normalizer, and added-token files.
- `preprocessor_config.json` and its 80-channel mel filter bank.

The feature extractor expects mono 16 kHz audio, pads or truncates to 30 seconds (480,000 samples), uses FFT length 400 and hop length 160, and produces `[1, 80, 3000]` log-Mel features. See the [upstream model repository](https://huggingface.co/openai/whisper-base), [Whisper feature-extractor documentation](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperFeatureExtractor), and the local [`feature_extraction_whisper.js`](../../packages/transformers/src/models/whisper/feature_extraction_whisper.js).

### Required execution semantics

A generic `forward({input_features})` is insufficient. The loaded backend model must expose `generate()` with Whisper behavior:

1. Run `encode` once for each 30-second feature window.
2. Build the decoder prompt from decoder-start, language, task, and timestamp policy tokens.
3. Maintain the fixed `[1, 128]` INT32 token buffer and the export's `[1, 1, 128, 128]` mask.
4. Run `decode` iteratively and select the vocabulary distribution at the current position.
5. Apply the configured forced/suppressed-token and end-of-sequence behavior.
6. Return generated token IDs in the form expected by the ASR pipeline.
7. Let `WhisperTokenizer._decode_asr()` merge chunks, strip prompt/control tokens, and produce text and optional timestamps.

The existing Transformers.js ASR pipeline performs chunking, feature extraction, `generate()`, stride conversion, and `_decode_asr()` orchestration. The LiteRT provider should not duplicate those outer pipeline responsibilities. See [`automatic-speech-recognition.js`](../../packages/transformers/src/pipelines/automatic-speech-recognition.js) and [`tokenization_whisper.js`](../../packages/transformers/src/models/whisper/tokenization_whisper.js).

The current Whisper Tiny LiteRT adapter is the closest implementation template because it already targets the same fixed decoder length and vocabulary size. Reuse should still be guarded by a numerical parity test for the Base export, especially for mask values and position selection; those details are inferred from the existing compatible export protocol rather than documented by the Base card.

The launch chart categorizes Whisper Base as a `Text Encoder`. Its latency should therefore be interpreted as encoder-subgraph latency unless benchmark code proves otherwise, not as end-to-end transcription latency.

## Ultralytics YOLO detection

### Why the DETR contract is incompatible

The current Transformers.js object-detection postprocessor expects separate `outputs.logits` and normalized center-format `outputs.pred_boxes`. It applies a softmax over classes, treats the last class as background, converts normalized boxes to corners, and rescales them to the original image. It does not perform NMS. See [`post_process_object_detection`](../../packages/transformers/src/image_processors_utils.js) and [`object-detection.js`](../../packages/transformers/src/pipelines/object-detection.js).

An Ultralytics one-to-many detection export instead returns one raw prediction tensor. In the inspected official post-launch YOLO26n W8A32 artifact:

- Input: FP32 `[1, 3, 640, 640]`, NCHW.
- Output: FP32 `[1, 84, 8400]`.
- Channels 0-3 are decoded center-x, center-y, width, and height in model-input pixels.
- Channels 4-83 are 80 independent COCO class probabilities. The exported head has already applied sigmoid; there is no separate objectness channel and no background class.
- The 8,400 candidates require confidence filtering and class-aware NMS outside the graph.

These semantics follow the official Ultralytics detection head, which concatenates decoded boxes with `scores.sigmoid()`, and its NMS implementation, which documents raw `[batch, 4 + classes, boxes]` inputs. See [`head.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py) and [`nms.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py).

Do not adapt this tensor by merely renaming it to DETR outputs. Softmax would corrupt independent class probabilities, the absent background channel would cause the last COCO class to be dropped, pixel-space boxes would be scaled as if normalized, and overlapping candidates would remain unsuppressed.

### Required preprocessing

Ultralytics detection preprocessing is RGB, aspect-preserving letterbox resize/pad to the export's input size, `1/255` rescaling, and CHW packing. The implementation must retain the resize ratio and left/top padding so final boxes can be mapped back to the original image. Padding convention and rounding must match the exporter/runtime recipe and should be tested on non-square images.

The current `onnx-community/yolo26n-ONNX` `YolosImageProcessor` sidecar is not sufficient by itself. Its fixed `{width: 640, height: 640}` resize uses the base processor's direct resize path and `do_pad = false`, which warps a non-square image rather than performing Ultralytics letterboxing. Its `config.json` does provide the standard 80-class COCO label order, which can be reused if the target artifact metadata confirms that class set. See the [config](https://huggingface.co/onnx-community/yolo26n-ONNX/blob/main/config.json), [processor config](https://huggingface.co/onnx-community/yolo26n-ONNX/blob/main/preprocessor_config.json), and [official COCO dataset config](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

### Required postprocessing

For a verified one-to-many `[1, 84, N]` export, the adapter must:

1. Transpose candidate access conceptually from channels-first to `N` rows.
2. Select the highest class probability per candidate, or support multi-label behavior explicitly.
3. Filter candidates using the requested confidence threshold without another sigmoid or softmax.
4. Convert `[cx, cy, width, height]` in input pixels to `[x1, y1, x2, y2]`.
5. Apply class-aware NMS with an explicit IoU threshold and maximum-detection limit.
6. Remove letterbox padding, divide by the resize ratio, and clip boxes to the original image dimensions.
7. Return `boxes`, `scores`, and integer class IDs in the shape expected by `ObjectDetectionPipeline`.

The existing pipeline exposes only `threshold` and `percentage`. Ultralytics-compatible behavior also needs an IoU threshold, maximum detections, optional class filtering, and optionally class-agnostic or multi-label NMS. Either extend the pipeline options or define documented adapter defaults. Note that `ObjectDetectionPipeline` defaults `threshold` to 0.9, while Ultralytics prediction commonly uses a substantially lower confidence threshold; users will otherwise observe far fewer detections even with correct decoding.

YOLO exports are not all interchangeable. An export may contain raw one-to-many output requiring NMS, an end-to-end/one-to-one output shaped like `[batch, max_detections, 6]`, or graph-integrated NMS. Register support only against a pinned artifact after validating input/output count, shape, embedded metadata, confidence activation, coordinate encoding, and NMS ownership. Ultralytics' current browser package makes the same distinction and recommends one-to-many exports for LiteRT.js WebGPU. See the [official browser README](https://github.com/ultralytics/inference/blob/main/web/README.md#-litertjs-backend).

## Implementation invariants

- Treat preprocessing, output naming, and postprocessing as descriptor/task semantics. LiteRT.js deliberately runs models but does not supply these pipeline stages; see the [LiteRT.js README](https://github.com/google-ai-edge/LiteRT/tree/main/litert/js#limitations).
- Validate runtime tensor counts, dtypes, and fixed dimensions immediately after compilation. Do not assume every model uses `serving_default`, NHWC, or an `Identity` output.
- Use the default LiteRT runner for the classifier models and named `encode`/`decode` runners for Whisper.
- Return CPU Transformers.js tensors with semantic names (`logits` for classifiers) after copying output data, then delete LiteRT input/output tensors in `finally` blocks.
- Keep raw INT8-I/O artifacts unsupported until the runtime exposes sufficient dtype and quantization scale/zero-point metadata. Weight-quantized files with FP32 external I/O do not have this problem.
- Pin tested model revisions and record expected hashes. Repository `main` revisions and conversion recipes can change independently of the runtime package.
- Add numerical fixture tests, not only shape tests: top-k ImageNet IDs for classifiers, token sequences for Whisper, and boxes/classes/scores on a non-square image for YOLO.

## Remaining uncertainties

- Google has not published enough information to reproduce or hash-match any exact launch artifact.
- The normalized chart's `efficientnet` label does not explicitly say B0; B0 is the strongest candidate from the linked LiteRT collection.
- The article does not explain why one detector figure says YOLO11n and the other says YOLO26n.
- The Whisper export card does not document the decoder mask's exact value convention or exporter source.
- The launch detector's output-head mode and postprocessing ownership are unknown. The inspected YOLO26n artifact establishes only one concrete post-launch contract.
- Exact numerical parity still requires browser runs through LiteRT.js on the selected WASM, WebGPU, and WebNN paths.
