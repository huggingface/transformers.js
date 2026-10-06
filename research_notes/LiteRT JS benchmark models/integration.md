# LiteRT.js benchmark model integration

## What assumptions in the current LiteRtInferenceProvider are Whisper-tiny-specific?

### Takeaway
Almost every policy decision after transport/runtime initialization is Whisper-tiny-specific. Preserve that behavior as a `whisper` execution adapter, but extract artifact selection, shared assets, task validation, and execution into descriptors so default-signature vision models do not inherit Whisper constants or generation behavior.

### Cited Findings
- `DEFAULT_SHARED_MODEL_ID`, `MAX_DECODE_LENGTH`, and `VOCAB_SIZE` are fixed to `openai/whisper-tiny`, 128 tokens, and 51,865 vocabulary entries. `selectModelFile()` recognizes only `whisper_tiny_30s_f32.tflite` and `whisper_tiny_30s_i8.tflite`. — [Current provider source](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/src/index.ts#L11-L44)
- `GenerationConfig`, `createCausalMask()`, `getPrompt()`, and `greedyGenerate()` encode Whisper-specific prompt tokens, language/task lookup, timestamp rejection, a fixed causal mask, `encode`/`decode` signatures, positional `args_0`/`args_1`/`args_2`, `output_0`, greedy suppression, and an `int64` token result. — [Current provider source](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/src/index.ts#L16-L25); [generation implementation](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/src/index.ts#L106-L203)
- Static capabilities advertise only `automatic-speech-recognition`, and `load()` rejects every other task. The returned model deliberately throws on a generic forward call and exposes only `generate()`; it declares no execution capability because speech recognition has no versioned capability contract yet. — [Current provider source](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/src/index.ts#L205-L215); [load implementation](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/src/index.ts#L264-L295)
- Artifact enumeration always includes `config.json`, one Whisper file, and `generation_config.json`; artifact metadata special-cases the default Whisper selection while accepting any `.tflite` suffix; deletion accepts every `.tflite` path. — [Current provider source](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/src/index.ts#L234-L262)
- The current README explicitly limits the provider to fixed-shape Whisper, says Transformers.js owns feature extraction/tokenization/ASR postprocessing, and documents the `q8`/`int8` to int8-file and `fp32` to float-file mapping. — [Package README](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/README.md#L5-L8); [package README](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/README.md#L25-L35)
- The Whisper Base repository documents the same named signatures and tensor protocol at a larger encoder width: `encode` maps `[1,80,3000]` to `[1,1500,512]`; `decode` consumes encoder output, `[1,128]` int32 tokens, and `[1,1,128,128]` mask and returns `[1,128,51865]`. Its files are `whisper_base_30s_f32.tflite` and `whisper_base_30s_i8.tflite`. — [Whisper Base model card](https://huggingface.co/litert-community/whisper-base/blob/main/README.md); [repository tree](https://huggingface.co/api/models/litert-community/whisper-base/tree/main?recursive=true&expand=false)
- Transformers.js generic backends may return a callable or a model with `forward()`, while ASR uses a loaded model's `generate()`. The image-classification pipeline calls `model({pixel_values})` and requires `output.logits`; the object-detection pipeline calls `model({pixel_values, pixel_mask})` and delegates output handling to the image processor. — [Inference backend contract](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/backends/inference.js#L1-L17); [image-classification pipeline](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/pipelines/image-classification.js#L82-L106); [object-detection pipeline](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/pipelines/object-detection.js#L60-L90)

### Inferences
- Keep the existing public class and `LiteRtInferenceProvider.from_modelId(artifactModelId, options)` signature. Resolve a descriptor in the constructor and keep current fallback behavior for `litert-community/whisper-tiny`; this is the least disruptive way to preserve existing behavior.
- Move the current `GenerationConfig`, `createCausalMask()`, `getPrompt()`, and `greedyGenerate()` code together into `src/tasks/whisper.ts`. Parameterize only descriptor facts that actually vary (`maxDecodeLength`, `vocabSize`, signature and tensor names). Whisper Base can then reuse the adapter with `sharedModelId: 'openai/whisper-base'` and different artifact filenames; its documented shapes show no algorithm change is needed.
- Remove Whisper policy from these symbols in `src/index.ts`: `DEFAULT_SHARED_MODEL_ID`, `selectModelFile`, class `capabilities`, `listModelArtifacts`, task validation in `load`, and callable construction. They should read the resolved descriptor instead.
- `readModelResponse()` and `hubUrl()` are unused because `load()` correctly delegates bytes to `options.getModelFile()`. Do not incorporate either into the new architecture; removing them in a later cleanup is safe but unnecessary for the benchmark extension.
- Do not change timestamp behavior, default English/transcribe prompting, suppression, token output type, or current dtype selection for Whisper Tiny. Those are compatibility-sensitive even if a future generic generation implementation would differ.

### Gaps
- The current code has no package-local tests, so preservation of Whisper Tiny behavior is not presently enforced. — No `packages/transformers-litertjs-runtime/tests` directory exists in the local checkout.
- The Whisper Base card lists the float file under its “Files” section but the repository tree also contains the int8 file; no card-level compatibility statement confirms which LiteRT.js accelerators successfully compile the int8 artifact. — [Model card](https://huggingface.co/litert-community/whisper-base/blob/main/README.md); [repository tree](https://huggingface.co/api/models/litert-community/whisper-base/tree/main?recursive=true&expand=false)

## What APIs does @litertjs/core 2.5.3 expose for unnamed/default signatures and model metadata?

### Takeaway
Version 2.5.3 has all APIs needed for a reusable default-signature executor: `CompiledModel` itself is the default `SignatureRunner`, and metadata is available through `getInputDetails()`/`getOutputDetails()`. It does not expose quantization scale/zero-point metadata, and its public host tensor dtypes are limited to float32, int32, and uint8.

### Cited Findings
- `SignatureRunner` exposes `key`, compile `options`, `getInputDetails()`, `getOutputDetails()`, and overloads of `run()` for one tensor, positional tensor arrays, or named tensor records. — [`@litertjs/core` 2.5.3 declarations](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L603-L625)
- `CompiledModel` implements `SignatureRunner`; its direct `run(input)` overloads run the default signature, while `run(signatureName, input)` runs an explicitly named signature. It also exposes `signatures`, `isFullyAccelerated`, and `delete()`. — [`@litertjs/core` 2.5.3 declarations](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L644-L693)
- Internally, 2.5.3 creates a runner for every model signature, keys the `signatures` record by each signature key, and chooses the first signature as `defaultSignature`. `CompiledModel.key`, input metadata, output metadata, and direct `run()` all delegate to that first signature. — [`@litertjs/core` 2.5.3 implementation](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.js#L816-L857)
- Named-record execution validates required names using `TensorDetails.name`; positional execution validates only tensor count. Returned values retain positional form for positional inputs and become a name-keyed record for record inputs. — [`@litertjs/core` 2.5.3 implementation](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.js#L664-L713)
- Each `TensorDetails` contains `name`, `index`, `dtype`, `shape`, and `supportedBufferTypes`. Shapes are exposed as `Int32Array`. — [`@litertjs/core` 2.5.3 declarations](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L603-L612)
- Runtime execution checks each input against the model tensor type and buffer requirements, automatically copies an input to a supported buffer type when necessary, and wraps output buffers as LiteRT tensors. — [`@litertjs/core` 2.5.3 implementation](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.js#L715-L779)
- Public LiteRT tensors can be constructed from typed arrays, expose `data()` and synchronous `toTypedArray()`, support copy/move, and require explicit `delete()`. — [`@litertjs/core` 2.5.3 declarations](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L487-L552)
- The package maps only `float32`/`Float32Array`, `int32`/`Int32Array`, and `uint8`/`Uint8Array` in its public `DType` and `TypedArray` types. — [`@litertjs/core` 2.5.3 declarations](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L394-L431)
- Compile options accept `accelerator: 'wasm' | 'webgpu' | 'webnn'`, plus GPU and WebNN options. The exported narrower `Accelerator` type itself contains only `webgpu` and `wasm`, explaining the current provider's local `Accelerator | 'webnn'` return type. — [`@litertjs/core` 2.5.3 declarations](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L65-L81); [`Accelerator` declaration](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L369-L376)
- Google's package README states that LiteRT.js is a browser runtime, demonstrates `loadLiteRt()`, `loadAndCompile()`, default `model.run()`, manual tensor deletion, and WASM/WebGPU/WebNN execution. It also states JSPI is required for WebNN and mixed WebGPU/WASM partitioning. — [`@litertjs/core` README](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/README.md#L32-L97)

### Inferences
- Add `src/session.ts` with a small `runDefaultSignature(compiled, descriptor, inputs)` function. Use positional `compiled.run(liteInputs)` based on descriptor `inputOrder`; this avoids coupling Transformers.js names such as `pixel_values` to converter-generated TFLite names while still checking count, dtype, and shape against `compiled.getInputDetails()`.
- Use named runners only for descriptors that require them. The Whisper adapter should continue using `compiled.signatures.encode` and `.decode`; a default vision model should use `compiled.run()` and never assume a signature key such as `serving_default`.
- Add `src/tensors.ts` with two narrowly scoped conversions:
  - `toLiteRtTensor(tensor, expectedDetails, transform?)`: validate supported Transformers.js type, apply an explicitly described layout transform if required, construct `new LiteRtTensor(data, dims)`, and let the caller delete it in `finally`.
  - `fromLiteRtTensor(tensor, outputName)`: await `tensor.data()`, derive type from `tensor.type.dtype`, derive dimensions from `tensor.type.layout.dimensions`, construct a CPU Transformers.js `Tensor`, then delete the LiteRT tensor after the copy.
- Return ordinary CPU Transformers.js tensors initially. Backend tensor storage integration is unnecessary for the benchmark models and would complicate ownership; LiteRT's automatic accelerator copy already handles host inputs.
- Validate metadata once immediately after compile. Error messages should include descriptor ID, expected semantic input/output name, runtime name, dtype, and shape. This catches incorrect artifacts before pipeline execution and makes otherwise opaque converted names diagnosable.
- Do not offer static-int8 ResNet as `int8`/`q8` yet. Its model card says both input and output are int8 and require stored quantization parameters, while LiteRT.js 2.5.3 exposes neither an `int8` public dtype nor scale/zero-point in `TensorDetails`. The dynamic weight-int8 artifact keeps float activations and is the safe compressed candidate.

### Gaps
- `TensorDetails` does not include quantization scale or zero-point. Therefore raw quantized model I/O cannot be converted correctly from metadata alone. — [Public metadata declaration](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L603-L612)
- The API documentation says every model has at least one signature, but it does not promise a stable key for an unnamed/default converted model; only direct `CompiledModel.run()` is safe without inspecting the artifact. — [`SignatureRunner` documentation](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L613-L625)
- Actual `key`, input names, output names, dtype, shape, and buffer requirements for each benchmark `.tflite` cannot be inferred from repository filenames or Python examples. They must be captured by compiling each artifact in a browser and logging `compiled.key`, `Object.keys(compiled.signatures)`, and all input/output details.

## How should multiple tasks be represented without per-model class duplication?

### Takeaway
Use immutable data descriptors plus three execution strategies: generic default-signature forward, Whisper generation, and (only once verified) YOLO detection adaptation. Keep transport, runtime initialization, compilation, artifact hooks, tensor conversion, and cleanup shared in one provider.

### Cited Findings
- The Transformers.js generic inference backend contract already separates a backend's shared `modelId`, static capabilities, artifact hooks, and `load()` from the loaded model's callable/`forward()`, `generate()`, capabilities, config, and disposal. — [Inference backend contract](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/backends/inference.js#L100-L134)
- `pipeline()` treats a non-session inference backend as a custom backend, validates its task list, asks its `listModelArtifacts()` implementation for backend files, and otherwise loads tokenizer/processor/config from the backend's shared `modelId`. — [`pipeline()` integration](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/pipelines.js#L194-L225)
- Shared assets can have backend-selected revision/subfolder settings; host-owned `fetch`, model-file loading, metadata, and deletion functions are passed across the backend boundary. — [Backend host/shared-asset helpers](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/backends/inference.js#L239-L277)
- Built-in normalized session providers require `sessionProvider.version === 1`, `constructSessions()`, and a `getSessionConfig()` result that maps architecture sessions. Transformers.js then instantiates a built-in model class around those sessions. — [Session-provider detection](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/backends/inference.js#L154-L169); [built-in model loading](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/models/modeling_utils.js#L437-L448)
- ONNX uses `getSessionConfig()` and `constructSessions()` because it supports many built-in architectures and normalized sessions with `inputNames`, `outputNames`, metadata, `run`, and `release`. — [ONNX provider](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-onnxruntime/src/provider.ts#L169-L175); [ONNX session construction](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-onnxruntime/src/provider.ts#L280-L320)
- The three confirmed classification repositories contain executable files but no `config.json` or `preprocessor_config.json`, so their Transformers.js config/labels/processor must come from a compatible shared-assets repository. — [EfficientNet B0 tree](https://huggingface.co/api/models/litert-community/efficientnet_b0/tree/main?recursive=true&expand=false); [Inception V3 tree](https://huggingface.co/api/models/litert-community/inception_v3/tree/main?recursive=true&expand=false); [ResNet50 tree](https://huggingface.co/api/models/litert-community/resnet50/tree/main?recursive=true&expand=false)
- EfficientNet B0 and ResNet50 examples use NCHW float32 input with ImageNet mean/std; Inception V3 uses NHWC float32 `[1,299,299,3]`. This is a descriptor-level layout difference, not a reason for separate provider classes. — [EfficientNet B0 model card](https://huggingface.co/litert-community/efficientnet_b0/blob/main/README.md); [ResNet50 model card](https://huggingface.co/litert-community/resnet50/blob/main/README.md); [Inception V3 model card](https://huggingface.co/litert-community/inception_v3/blob/main/README.md)
- EfficientNet B0 offers float, static mixed-int8, weight-only int8/float-activation, and a device-plugin artifact; ResNet50 offers float, dynamic int8-weight/float-activation, and static int8 I/O; Inception currently exposes float and a device-plugin artifact. — [EfficientNet B0 tree](https://huggingface.co/api/models/litert-community/efficientnet_b0/tree/main?recursive=true&expand=false); [ResNet50 tree](https://huggingface.co/api/models/litert-community/resnet50/tree/main?recursive=true&expand=false); [Inception V3 tree](https://huggingface.co/api/models/litert-community/inception_v3/tree/main?recursive=true&expand=false)
- The current object-detection postprocessor requires `outputs.logits` shaped `[batch, boxes, classes]` and `outputs.pred_boxes`, applies softmax with a final background class, and treats boxes as normalized center coordinates. — [Transformers.js object-detection postprocessor](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/image_processors_utils.js#L69-L146)
- The available Transformers.js YOLO26 ONNX shared-assets candidate uses `model_type: 'yolos'`, COCO labels, and a 640x640 rescale-only `YolosImageProcessor`, but the queried `litert-community` organization does not currently return a YOLO26n model. — [YOLO26 ONNX config](https://huggingface.co/onnx-community/yolo26n-ONNX/blob/main/config.json); [preprocessor config](https://huggingface.co/onnx-community/yolo26n-ONNX/blob/main/preprocessor_config.json); [LiteRT-community YOLO search](https://huggingface.co/api/models?author=litert-community&search=yolo&limit=100)

### Inferences

#### Recommended files and symbols

- `src/descriptors.ts`
  - `type LiteRtModelDescriptor`
  - `const MODEL_DESCRIPTORS`
  - `resolveModelDescriptor(artifactModelId, options)`
  - Descriptor fields: `id`, `artifactModelId`, `sharedModelId`, `task`, `artifacts`, `execution`, `inputOrder`, `outputOrder`, `inputLayouts`, optional `metadataAssertions`, and optional `sharedAssets`.
- `src/artifacts.ts`
  - `selectArtifact(descriptor, dtype, device, modelFileOverride)`
  - `listDescriptorArtifacts(descriptor, options)`
  - Shared implementations for `getModelArtifactMetadata` and `deleteModelArtifact`, parameterized by the selected descriptor/artifact rather than suffix alone.
- `src/runtime.ts`
  - `ensureLiteRtLoaded(wasmPath, jspi)` retaining the current `getGlobalLiteRtPromise()` behavior.
  - `compileDescriptorModel(bytes, options)` retaining `loadAndCompile(bytes, {accelerator, ...session_options})` and `selectAccelerator()`.
- `src/tensors.ts`
  - `toLiteRtTensor`, `fromLiteRtTensor`, and one explicit NCHW-to-NHWC transform used only when a descriptor requests it.
- `src/session.ts`
  - `createDefaultForwardModel(compiled, descriptor, config)` returning a callable whose result keys come from descriptor semantic output names, not converter-generated runtime names.
  - `runDefaultSignature(compiled, descriptor, inputs)` with all input/output deletion in `try/finally`.
- `src/tasks/whisper.ts`
  - Existing `GenerationConfig`, `createCausalMask`, `getPrompt`, and `greedyGenerate`, with current defaults preserved.
  - `createWhisperModel(compiled, descriptor, generationConfig)` returning the same throwing forward callable, `generate`, ASR capability, and `dispose` behavior.
- `src/tasks/detection.ts`
  - Add only after YOLO output semantics are known. Prefer `adaptYoloOutputs(raw, descriptor)` as a function selected by a descriptor, not a YOLO provider subclass.
- `src/index.ts`
  - Continue exporting `LiteRtInferenceProvider` and `LiteRtInferenceProviderOptions`.
  - Keep `from_modelId()` and constructor source-compatible.
  - Provider fields delegate to the resolved descriptor; `load()` performs shared runtime/artifact work and dispatches on `descriptor.execution`.

#### Descriptor shape

```ts
type LiteRtModelDescriptor = {
    id: string;
    sharedModelId: string;
    tasks: readonly string[];
    artifacts: Partial<Record<'fp32' | 'int8' | 'q8', string>>;
    execution: 'default-forward' | 'whisper' | 'yolo-detection';
    inputOrder: readonly string[];
    outputOrder: readonly string[];
    inputLayouts?: Readonly<Record<string, 'nchw' | 'nhwc'>>;
    optionalConfigs?: readonly string[];
};
```

- Keep descriptors data-only. Execution strategy functions should be shared by task; no `EfficientNetProvider`, `InceptionProvider`, `ResNetProvider`, or `WhisperBaseProvider` classes.
- Initial descriptor intent, pending metadata capture:
  - `whisper-tiny`: current filenames, `openai/whisper-tiny`, ASR, named Whisper execution.
  - `whisper-base`: base filenames, `openai/whisper-base`, ASR, the same named Whisper execution.
  - `efficientnet_b0`: float and weight-only artifacts, image classification, default forward, NCHW input, semantic output `logits`.
  - `resnet50`: float and dynamic-weight-int8 artifacts, image classification, default forward, NCHW input, semantic output `logits`.
  - `inception_v3`: float artifact only initially, image classification, default forward, NHWC input, semantic output `logits`.
  - `YOLO26n`: do not register as supported until artifact repository, filename, default signature metadata, raw output layout, box encoding, confidence activation, and NMS ownership are confirmed.
- Provider construction should remain `LiteRtInferenceProvider.from_modelId('litert-community/efficientnet_b0')`. Descriptor lookup can use exact repository IDs with a final path-segment alias only if aliases are unambiguous. `options.sharedModelId` and `options.modelFile` remain expert overrides and therefore preserve the current Whisper Tiny API.
- Keep the generic backend path. A normalized session provider is not the smallest architecture here: Whisper already has a custom generation callable; Inception has no corresponding built-in Transformers.js model class in the local registry; and YOLO output adaptation is not a generic normalized-session concern. The session-like executor should stay internal to this package.
- Artifact hooks must use `artifactModelId`, descriptor-selected filenames, `revision`, and `subfolder: null`, as the current implementation does. `listModelArtifacts()` should include `config.json` for shared assets and include `generation_config.json` only for Whisper descriptors.
- `capabilities` must be descriptor-derived per provider instance. A union of all tasks on every instance would defeat early pipeline validation.

### Gaps
- No exact YOLO26n LiteRT artifact ID or filename was supplied, and no `litert-community/YOLO26n` repository was found. `software-zetic/YOLO26n` currently contains only `.gitattributes`, so it cannot establish a runnable artifact contract. — [Repository tree](https://huggingface.co/api/models/software-zetic/YOLO26n/tree/main?recursive=true&expand=false)
- YOLO26 raw output semantics are unknown. Safe implementation requires the actual output count/names/shapes and confirmation of whether decoding and NMS are inside the graph. Without that, mapping an output to DETR-style `logits`/`pred_boxes` would likely be behaviorally wrong.
- A compatible shared-assets ID is not established for each torchvision-converted classifier. The LiteRT cards document preprocessing, but their repositories do not ship Transformers.js config/processor files. In particular, `google/efficientnet-b0`'s published processor values differ from the torchvision preprocessing described by the LiteRT EfficientNet card, so it should not be selected without an accuracy check. — [Google EfficientNet processor](https://huggingface.co/google/efficientnet-b0/blob/main/preprocessor_config.json); [LiteRT EfficientNet card](https://huggingface.co/litert-community/efficientnet_b0/blob/main/README.md)
- Inception is absent from the local Transformers.js image-classification model registry, reinforcing the generic-backend choice, but a complete shared config with ImageNet labels and a Transformers.js-recognized image processor still needs to be identified or published. — [Local registry](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/models/registry.js#L425-L448)
- Device compatibility is not uniform across files. Descriptor artifact selection needs an explicit supported-device matrix rather than choosing solely by dtype; device-plugin files and static-int8 files must not be selected implicitly.

## What can be tested without a browser and what requires mocked compiled models?

### Takeaway
Most provider policy and all execution/ownership logic can be unit-tested in Node if `@litertjs/core` is mocked. Real compilation, metadata discovery, delegates, JSPI, and numerical compatibility need browser integration tests against the actual artifacts.

### Cited Findings
- `@litertjs/core` describes itself as a core web runtime and its own test script uses `jasmine-browser-runner`; runtime WASM assets must be served to the browser. — [`@litertjs/core` package](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/package.json#L22-L29); [`@litertjs/core` README](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/README.md#L3-L16)
- The current runtime package has build/typegen scripts but no test script or test dependencies. — [Runtime package manifest](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/package.json#L14-L31)
- The ONNX provider tests demonstrate the local convention for testing provider identity, artifact resolution, host-service forwarding, and per-load cache-aware model loading with Jest mocks, without running a real inference runtime. — [ONNX provider tests](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-onnxruntime/tests/provider.test.js#L1-L61); [host-service tests](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-onnxruntime/tests/provider.test.js#L63-L161)
- Transformers.js owns model-file metadata and deletion hooks for custom backends; `get_file_metadata()` first asks the backend and falls back to shared assets, while cache clearing asks the backend to delete backend-owned files. — [Metadata integration](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/utils/model_registry/get_file_metadata.js#L59-L82); [cache deletion integration](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers/src/utils/model_registry/clear_cache.js#L58-L94)
- LiteRT tensors and compiled models require explicit deletion, and the runner may create temporary accelerator copies. — [`Tensor.delete()` declaration](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.d.ts#L503-L552); [runner cleanup](file:///Users/nico/Documents/Dev/transformers.js/packages/transformers-litertjs-runtime/node_modules/@litertjs/core/dist/index.js#L664-L779)

### Inferences

#### Node tests without a compiled-model mock

- Add `tests/descriptors.test.js` for exact descriptor resolution, unknown-model errors, per-instance task/device/dtype capabilities, shared model IDs, Whisper Tiny backward compatibility, and option overrides.
- Add `tests/artifacts.test.js` for model-file selection by dtype/device, optional `generation_config.json`, unsupported combinations, model-file override, and exact artifact lists for all confirmed descriptors.
- Add `tests/provider.test.js` for `from_modelId()`, task rejection, `getModelFile()` arguments, metadata delegation, deletion delegation, revision handling, `subfolder: null`, invalid byte responses, and disposal delegation. Follow the ONNX provider's Jest style.
- Add pure tests for accelerator selection and NCHW-to-NHWC permutation using small arrays. Include shape mismatch, unsupported dtype, and unknown semantic input/output errors.
- Test that `options.session_options` cannot accidentally replace the descriptor-selected accelerator unless preserving current spread precedence is intentional. The current code spreads `session_options` after `accelerator`, so callers can override it.

#### Node tests with mocked LiteRT tensors and compiled models

- Mock `loadLiteRt`, `getGlobalLiteRtPromise`, `loadAndCompile`, and `Tensor` from `@litertjs/core`. A fake compiled model should expose `key`, `signatures`, `getInputDetails`, `getOutputDetails`, `run`, and `delete`.
- Default-forward cases: semantic input ordering, positional runtime invocation, output ordering to `logits`, dtype/dim conversion, NHWC transform, ignored `pixel_mask` only when the descriptor explicitly excludes it, runtime errors, and deletion of every input/output tensor on success and failure.
- Metadata cases: wrong count, wrong rank, wrong fixed dimension, unsupported runtime dtype, and informative descriptor-specific errors.
- Whisper regression cases: `encode`/`decode` invocation names and ordering, Tiny constants, Base encoder width accepted from metadata, prompt defaults, language/task errors, timestamp rejection, suppression at first/all steps, EOS termination, max token clamp, output `int64` dimensions, and cleanup when encode/decode/data throws.
- Artifact-host cases: fake `getModelFile`, `getModelFileMetadata`, and `deleteModelFile` should assert use of `artifactModelId`, while loaded `config` and processors continue to belong to `sharedModelId`.
- A future YOLO adapter needs fixture tensors captured from the real model. Unit tests should verify its decode math, coordinate convention, confidence handling, class IDs, threshold behavior, and NMS independently of LiteRT.

#### Browser integration tests

- Serve the package's WASM directory and compile every selected artifact on WASM. Record and assert default/named signatures plus complete input/output metadata.
- Run one known fixture per model and compare output shape and a small numerical/top-k golden tolerance. For Whisper Tiny and Base, compare a short transcription token sequence; for classifiers, compare top-k class IDs; for YOLO, compare decoded boxes/classes/scores.
- Run WebGPU and WebNN smoke tests separately, including `isFullyAccelerated`, expected fallback behavior, and JSPI configuration. Do not infer support from the provider capability list alone.
- Exercise abort/download/progress/cache behavior in a browser because real `Response.body`, WASM loading, and browser cache semantics are not represented by compiled-model mocks.
- Include repeated load/run/dispose cycles and failure injection to detect WASM/GPU resource leaks.

### Gaps
- The monorepo currently has no configured test command in the LiteRT package. Before adding tests, decide whether they run through the root package test harness or whether this package should gain Jest configuration and a `test` script.
- Mocking an ESM dependency may require the same Jest transform/module-mapping conventions as the rest of the monorepo; those conventions are not present in this new package yet.
- Browser numerical goldens cannot be authored safely until the exact artifact revisions and signature metadata are pinned. Descriptors should ideally pin or document tested revisions even if the public option continues to default to `main`.
