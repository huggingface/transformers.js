# Integrate an inference runtime with Transformers.js

This guide is for runtime authors who want Transformers.js to execute models with their own inference engine. Your integration is an **inference provider**: a small adapter that loads your runtime, converts tensors, executes the model, and releases native resources.

Transformers.js continues to provide the user-facing APIs and model semantics:

- `pipeline()` and `AutoModel.from_pretrained()`;
- configuration, tokenizer, processor, and generation-config loading;
- text, image, audio, and video preprocessing;
- pipeline postprocessing and output formatting;
- Hub authentication, progress events, cancellation, and model-file caching.

Your provider owns runtime-specific behavior:

- selecting the executable artifact for a device and data type;
- initializing or compiling your runtime session;
- converting Transformers.js tensors to and from runtime-native tensors;
- executing the model;
- releasing sessions, buffers, and other native resources.

The provider contract is structural. You do not need to extend a Transformers.js class or import private Transformers.js modules.

## Contract stability

The contract has two tiers. Build on the stable tier. Use the experimental tier only when you are prepared to follow changes in minor releases.

**Stable.** These parts will only change in a major release:

- the provider object: `modelId`, `load(options)`, static `capabilities`, `sharedAssets`, `chatTemplate`, and the registry hooks `listModelArtifacts()`, `getModelArtifactMetadata()`, `deleteModelArtifact()`;
- the host services passed to `load()`: `getModelFile()`, `getModelFileMetadata()`, `deleteModelFile()`, and `fetch`;
- the loaded model: `forward(inputs)`, `dispose()`, and the `forward` capability;
- device-resident tensors: `Tensor.fromBackendStorage()`, `tensor.getBackendStorage()`, and `TensorOpRegistry`.

**Experimental.** These parts are being developed together with a first external runtime. They are exported and typed, and they work today, but their shape may change in a minor release:

- the normalized-session contract (`sessionProvider` and `constructSessions()`);
- causal text generation (`capabilities.causalGeneration` and `createAutoregressiveSession()`);
- the object-detection hooks (`capabilities.objectDetection`);
- the random-access `artifactProvider` loading option.

Each experimental part is marked `@experimental` in the Transformers.js type definitions and described in [Experimental contracts](#experimental-contracts) below. The portable execution manifest that will eventually replace the ONNX-shaped session roles is a proposal, not a contract; see [INFERENCE_EXECUTION_MANIFEST_PROPOSAL.md](INFERENCE_EXECUTION_MANIFEST_PROPOSAL.md).

## 1. Implement the generic forward contract

For an ordinary model, implement the generic forward contract. Transformers.js gives the loaded model named `Tensor` inputs and expects named `Tensor` outputs:

```ts
type InferenceModel = {
    forward(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    dispose(): Promise<unknown> | unknown;
    capabilities?: {
        forward?: { version: 1 };
    };
};
```

This is the recommended starting point. It lets Transformers.js retain preprocessing, postprocessing, and task semantics while your package only handles execution. A model may also be a plain callable function with `dispose()` attached; Transformers.js treats a callable and a `forward()` method the same way.

Transformers.js reads exactly three loaded-model capabilities: `forward`, `causalGeneration`, and `objectDetection`. Declare only the ones your model implements. Do not invent capability names or copy unversioned methods from another provider and expect Transformers.js to detect them. If a task needs an execution contract that does not exist yet, propose the versioned contract in Transformers.js first.

## 2. Implement the provider

A generic forward provider only requires:

- `modelId`: the repository ID or local path containing the Transformers.js-compatible shared assets;
- `load(options)`: a function that returns the executable model.

`modelId` normally points to a repository containing `config.json` plus the tokenizer or processor files for the model. The runtime artifact may live in this repository or in a separate repository known by your provider.

The following provider is a complete outline for a runtime with a generic forward pass:

```js
import { Tensor } from '@huggingface/transformers';
import { createSession, RuntimeTensor } from 'my-inference-runtime';

export class MyInferenceProvider {
    capabilities = {
        devices: ['cpu', 'webgpu'],
        dtypes: ['fp32', 'int8'],
        tasks: ['text-classification'],
    };

    constructor(
        modelId,
        {
            artifactModelId = modelId,
            artifactRevision = 'main',
            modelFile = 'model.bin',
        } = {},
    ) {
        this.modelId = modelId;
        this.artifactModelId = artifactModelId;
        this.artifactRevision = artifactRevision;
        this.modelFile = modelFile;
    }

    selectModelFile(_options) {
        return this.modelFile;
    }

    artifactOptions(options) {
        return { ...options, revision: this.artifactRevision, subfolder: null };
    }

    async load(options) {
        this.validateOptions(options);
        options.signal?.throwIfAborted();

        const modelFile = this.selectModelFile(options);
        const modelData = await options.getModelFile(
            this.artifactModelId,
            modelFile,
            this.artifactOptions(options),
        );
        const session = await createSession(modelData, {
            device: options.device,
            dtype: options.dtype,
            ...options.session_options,
        });

        return {
            capabilities: { forward: { version: 1 } },

            async forward(inputs) {
                const runtimeInputs = Object.fromEntries(
                    Object.entries(inputs).map(([name, tensor]) => [
                        name,
                        new RuntimeTensor(
                            tensor.type,
                            tensor.data,
                            tensor.dims,
                        ),
                    ]),
                );
                const outputs = await session.run(runtimeInputs);
                return Object.fromEntries(
                    Object.entries(outputs).map(([name, tensor]) => [
                        name,
                        new Tensor(tensor.type, tensor.data, tensor.dims),
                    ]),
                );
            },

            dispose() {
                return session.dispose();
            },
        };
    }

    validateOptions(options) {
        if (
            options.device &&
            !this.capabilities.devices.includes(options.device)
        ) {
            throw new Error(`Unsupported device: ${options.device}`);
        }
        if (
            options.dtype &&
            !this.capabilities.dtypes.includes(options.dtype)
        ) {
            throw new Error(`Unsupported dtype: ${options.dtype}`);
        }
    }
}
```

The example assumes that `createSession()` accepts the value returned by `getModelFile()`. Adapt that boundary to your runtime. In browsers, model files are returned as `Uint8Array`. Node providers that require a filesystem path set `returnPath: true` in the options:

```js
const bytesOrPath = await options.getModelFile(modelId, file, {
    ...options,
    returnPath: isNode,
});
```

Keep runtime-native tensors inside the provider. Public inputs and outputs must use the Transformers.js `Tensor` type and preserve the model's input and output names.

## 3. Keep tensors in provider-owned storage

Use `Tensor.fromBackendStorage()` when a runtime output should retain its native handle instead of being copied to CPU:

```js
import { Tensor } from '@huggingface/transformers';

const output = Tensor.fromBackendStorage({
    backend: 'my-webgpu-runtime',
    handle: gpuBuffer,
    type: 'float32',
    dims: [1, 384],
    size: 384,
    location: 'gpu-buffer',
    get data() {
        throw new Error('This tensor must be materialized before CPU access.');
    },
    dispose() {
        gpuBuffer.destroy();
    },
});
```

`backend` is the stable owner identifier used for handle interoperability and tensor-operation dispatch. `handle` is opaque to Transformers.js and must only be consumed after `tensor.getBackendStorage(expectedBackend)` returns a value. The declared `size` must match `dims`, and `dispose()` is invoked at most once by `Tensor.dispose()`.

Device storage does not need to provide synchronous CPU data. Its `data` accessor may throw an actionable error. Tensor methods that require `.data` remain a materialization boundary, so a provider should register operations for postprocessing that it wants to keep on device.

Register operations with the same backend identifier:

```js
import { TensorOpRegistry } from '@huggingface/transformers';

TensorOpRegistry.register('my-webgpu-runtime', {
    async mean_pooling(hiddenState, attentionMask) {
        return runMeanPoolingKernel(hiddenState, attentionMask);
    },
    async slice_tensor(input, ...slices) {
        return runSliceKernel(input, slices);
    },
    async normalize(input, p, dim) {
        return runNormalizeKernel(input, { p, dim });
    },
});
```

The feature-extraction pipeline dispatches mean pooling, token slicing, and normalization by tensor backend. If no backend implementation is registered, it uses the existing CPU implementation. Quantization still requires CPU-visible data. Providers should unregister operations during package teardown or tests with `TensorOpRegistry.unregister(backend, implementation)`.

Generic asynchronous tensor operations such as matrix multiplication, interpolation, FFT, top-k, and graph slicing also select their implementation from the non-CPU input tensor's backend. Inputs from two different non-CPU backends are rejected rather than interpreting one backend's opaque handle as another backend's tensor.

## 4. Use host-provided loading services

Transformers.js passes resolved pretrained options and host services to `load()`:

```ts
type InferenceBackendLoadOptions = PretrainedModelOptions & {
    modelId: string;
    fetch: typeof globalThis.fetch;
    getModelFile: (
        modelId: string,
        file: string,
        options?: PretrainedModelOptions & {
            fatal?: boolean; // default true: reject instead of resolving null
            returnPath?: boolean; // default false: return a path where supported
        },
    ) => Promise<string | Uint8Array | null>;
    getModelFileMetadata: (
        modelId: string,
        file: string,
        options?: PretrainedModelOptions,
    ) => Promise<{ exists: boolean; size?: number; fromCache?: boolean }>;
    deleteModelFile: (
        modelId: string,
        file: string,
        options?: PretrainedModelOptions,
    ) => Promise<boolean>;
    task?: string;
    config?: PretrainedConfig;
    generation_config?: Record<string, unknown>;
    artifactMetadata?: Record<string, { size?: number; fromCache?: boolean }>;
};
```

All three file services share one shape: the repository ID, the file name, and an options object. Pass the load options through and override only what differs for your artifact, usually `revision` and `subfolder`. The options also include normal loading values such as `device`, `dtype`, `revision`, `subfolder`, `cache_dir`, `local_files_only`, `session_options`, `progress_callback`, and `signal`.

Always load model artifacts with `options.getModelFile()`. Do not construct a Hub URL and call `globalThis.fetch()` for model files. The host loader provides:

- browser and filesystem cache integration;
- authentication and the configured Transformers.js transport;
- progress callbacks and known artifact sizes;
- request deduplication;
- `local_files_only` behavior;
- abort-signal handling.

Use `options.fetch` instead of `globalThis.fetch` for network requests that are not model artifacts. Check `options.signal` before and after expensive asynchronous work. If loading fails, release every session or buffer that was already created before rethrowing the error.

These options are the complete public host surface. The built-in ONNX provider additionally receives a private host object with the Transformers.js environment, logger, and cache because it is a first-party package; third-party providers do not, and should not model their integration on it.

## 5. Describe supported configurations

Add static provider capabilities so applications can inspect support before loading:

```js
capabilities = {
    devices: ['cpu', 'webgpu'],
    dtypes: ['fp32', 'int8'],
    tasks: ['text-classification', 'feature-extraction'],
};
```

The `tasks` list enables early pipeline validation. Device and dtype values are advisory because availability can still depend on the browser, operating system, runtime version, or model. Validate the selected combination again in `load()` and return an actionable error.

Provider capabilities and loaded-model capabilities serve different purposes. Provider capabilities describe what might be loadable. Loaded-model capabilities identify the execution contracts implemented by that particular model instance.

## 6. Support separate artifact repositories

It is common to keep runtime-specific binaries in one repository and shared Transformers.js assets in another. In that case:

- set `modelId` to the repository containing compatible config, tokenizer, and processor assets;
- keep the runtime repository ID in a provider property such as `artifactModelId`;
- pass `artifactModelId` to `getModelFile()`, `getModelFileMetadata()`, and `deleteModelFile()`;
- use identical file-selection logic for loading and artifact discovery.

You can pin shared assets to a revision or subfolder:

```js
provider.sharedAssets = {
    revision: 'refs/pr/42',
    subfolder: 'transformers-assets',
};
```

`sharedAssets` changes only the `revision` and `subfolder` used for shared files. It does not change `modelId` or describe the runtime artifact repository.

Be aware of what this split can and cannot do today. A provider that borrows shared assets from a different repository must know, per model, which repository is compatible and what the runtime graph's input and output names and shapes are. The experimental LiteRT.js provider does this with a hard-coded descriptor table. That is the right approach for a curated list of fixed-shape models. A provider that wants to run arbitrary Hub repositories without per-model work needs the experimental normalized-session contract described below.

## 7. Integrate with `ModelRegistry` and cache management

The basic provider works without registry hooks. Add the following hooks so applications can list required files, inspect download sizes and cache state, and clear runtime artifacts:

```ts
type InferenceBackend = {
    listModelArtifacts?(
        options: object,
    ): readonly string[] | Promise<readonly string[]>;
    getModelArtifactMetadata?(
        file: string,
        options: object,
    ): Promise<{ size?: number; fromCache?: boolean } | null>;
    deleteModelArtifact?(file: string, options: object): Promise<boolean>;
};
```

When your artifact uses the Transformers.js cache, delegate metadata and deletion to the host:

```js
listModelArtifacts(options) {
    return [this.selectModelFile(options)];
}

async getModelArtifactMetadata(file, options) {
    if (file !== this.selectModelFile(options)) return null;
    const metadata = await options.getModelFileMetadata(
        this.artifactModelId,
        file,
        this.artifactOptions(options),
    );
    return metadata.exists
        ? { size: metadata.size, fromCache: metadata.fromCache }
        : null;
}

async deleteModelArtifact(file, options) {
    if (file !== this.selectModelFile(options)) return false;
    return options.deleteModelFile(
        this.artifactModelId,
        file,
        this.artifactOptions(options),
    );
}
```

Return `null` from `getModelArtifactMetadata()` when the file is not owned by the provider. This allows Transformers.js to resolve it as a normal shared file. Return `false` from `deleteModelArtifact()` when your provider did not delete the file.

`listModelArtifacts()` and `load()` must select exactly the same artifacts for every task, device, data type, revision, and provider option. A mismatch causes incorrect cache state, progress totals, or incomplete cache deletion.

If your runtime has its own cache instead, implement metadata and deletion against that cache. The hook contract is about observable cache behavior; it does not require Transformers.js storage.

## 8. Configure chat templates when needed

A provider can select a default tokenizer chat template. Use inline content:

```js
provider.chatTemplate = {
    content: "{{ messages | map(attribute='content') | join('\\n') }}",
};
```

Or reference a file:

```js
provider.chatTemplate = {
    modelId: 'organization/prompt-assets',
    revision: 'main',
    subfolder: 'templates',
    file: 'chat_template.jinja',
};
```

Omitted file-source values default to the provider's `modelId`, selected revision and subfolder, and `chat_template.jinja`. Transformers.js resolves the template and installs it on the pipeline tokenizer.

## 9. Load the provider in an application

Pass a provider instance anywhere the API accepts a model:

```js
import { pipeline } from '@huggingface/transformers';
import { MyInferenceProvider } from '@organization/my-transformers-runtime';

const provider = new MyInferenceProvider('organization/model-assets', {
    artifactModelId: 'organization/model-my-runtime',
    modelFile: 'model.int8.bin',
});

const classifier = await pipeline('text-classification', provider, {
    device: 'webgpu',
    dtype: 'int8',
    progress_callback: (event) => console.log(event),
});

const result = await classifier('Transformers.js can use my runtime.');
await classifier.dispose();
```

A class with static `modelId` and `load()` members is also valid, but provider instances are usually better when users need to select artifact repositories, runtime initialization settings, or model files.

## 10. Test the complete integration

Test the provider at the following levels:

1. Test artifact selection for every supported device, dtype, task, and custom option.
2. Test tensor conversion, named outputs, failure cleanup, abort handling, and idempotent disposal.
3. Run a real `pipeline()` so the test covers shared-asset loading, preprocessing, execution, and postprocessing together.
4. Exercise `ModelRegistry.get_model_files()`, metadata lookup, cache detection, and cache clearing when registry hooks are implemented.
5. Verify the first load downloads artifacts and a second load reuses the cache in both browser and Node environments supported by the runtime.

Use [`packages/transformers-litertjs-runtime`](packages/transformers-litertjs-runtime) as a compact example of a separate runtime repository with host-managed caching. [`packages/transformers-onnxruntime`](packages/transformers-onnxruntime) is the first-party default provider; it shows how the normalized-session contract maps many architectures to multiple runtime sessions, but it also uses private host services that are not part of the public contract.

## Experimental contracts

Everything in this section works today and is covered by tests, but is marked `@experimental` in the type definitions. Each contract is being shaped with a first external runtime and may change in a minor release. If you build on one, pin your Transformers.js version and follow the changelog.

### Normalized sessions: one provider for all built-in architectures

To reuse the built-in Transformers.js model classes across many architectures, implement the normalized-session contract instead of a model-specific `load()`. A session provider needs `modelId`, `sessionProvider: { version: 1 }`, and `constructSessions()`:

```js
export class MySessionProvider {
    sessionProvider = { version: 1 };

    constructor(modelId) {
        this.modelId = modelId;
    }

    async constructSessions(names, options, cacheSessions) {
        return Object.fromEntries(
            await Promise.all(
                Object.entries(names).map(async ([role, file]) => {
                    const modelData = await options.getModelFile(
                        this.modelId,
                        file,
                        options,
                    );
                    const session = await compileGraph(modelData, options);
                    return [
                        role,
                        {
                            inputNames: session.inputNames,
                            outputNames: session.outputNames,
                            inputMetadata: session.inputMetadata,
                            outputMetadata: session.outputMetadata,
                            config: session.config,
                            run: (inputs) => runSession(session, inputs),
                            release: () => session.dispose(),
                        },
                    ];
                }),
            ),
        );
    }
}
```

Transformers.js selects the semantic model class from `config.json`, determines the required session roles, and calls `constructSessions()`. The provider compiles those graph files and returns normalized sessions. Each returned session must provide `inputNames`, `outputNames`, `run(inputs)`, and `release()`. Metadata and runtime-specific `config` are optional. If one session fails to initialize, release every session already created before rejecting `constructSessions()`.

This path lets one graph runtime reuse architecture-specific forward methods, generation orchestration, and output classes without implementing a loader for every model class. It is the answer for a runtime that can execute an exported op graph, for example a WebGPU engine that parses ONNX files and runs them with its own kernels.

Know the limitation before you build on it. The session roles and base file names in `names` come from the built-in ONNX session configuration, so they follow ONNX export conventions such as `model`, `encoder_model`, or `decoder_model_merged`. A non-ONNX runtime has to map those conventions onto its own artifacts. The [execution manifest proposal](INFERENCE_EXECUTION_MANIFEST_PROPOSAL.md) describes how this is meant to become portable; until it is implemented, the contract is experimental.

### Causal text generation

A model that owns its decoding loop declares `capabilities.causalGeneration` and implements `createAutoregressiveSession(options)`. Transformers.js then installs its own `generate()` on the model and drives the session, so the public generation API, logits processors, stopping criteria, and streaming behave the same as for built-in models.

The session protocol has two flavors. The pull session is the minimum: `prefill(inputs)` and `decode(inputs)` each return a logits lease that Transformers.js reads and releases, and Transformers.js samples on the CPU. The plan session is additive: `generateWithPlan(inputs, plan)` lets a runtime that keeps logits on device run sampling and token-mask processors itself. The capability object tells Transformers.js which modes, batch sizes, and pipeline depths the session supports.

See `CausalGenerationCapabilitiesV1`, `PullAutoregressiveSessionV1`, `PlanAutoregressiveSessionV1`, and `LogitsLeaseV1` in the type definitions for the exact shapes. No provider in this repository implements this contract yet; it is being developed against an external WebGPU runtime.

### Object-detection hooks

Some fixed-shape detection models bake preprocessing or postprocessing into the graph or require model-specific decoding of the raw output. A loaded model can take over either step by declaring `capabilities.objectDetection` with `preprocess: 'model'` and/or `postprocess: 'model'` and implementing `preprocessObjectDetection(images)` and/or `postProcessObjectDetection(outputs, threshold, targetSizes)`. The object-detection pipeline calls those methods in place of its own steps. The LiteRT.js YOLO model uses this hook.

This is a task-specific escape hatch. It will either be generalized into one mechanism for model-owned pipeline steps or replaced; do not use it as a template for other tasks.

### Random-access artifact provider

Applications can pass an `artifactProvider` loading option with `readJson(file)` and `openByteSource(file)`. A byte source supports concurrent, independently positioned range reads, which lets a runtime that parses graph files itself read weights lazily instead of loading whole files into memory. Transformers.js validates the provider's shape and passes it through to `load()` unchanged; it does not read from it. See `InferenceArtifactProvider` and `RandomAccessByteSource` in the type definitions.

## Integration checklist

- The provider exposes a string `modelId` plus `load(options)`, or the experimental normalized-session contract.
- `load()` uses `options.getModelFile()` for every runtime artifact.
- Network requests use the host-provided `options.fetch`.
- Loading honors `device`, `dtype`, `session_options`, `local_files_only`, and `signal` where applicable.
- The loaded model accepts and returns named Transformers.js tensors.
- The loaded model declares only the execution capabilities Transformers.js reads: `forward`, and where applicable the experimental `causalGeneration` or `objectDetection`.
- `dispose()` releases all runtime resources.
- Static capabilities accurately describe supported tasks, devices, and data types.
- Artifact discovery and artifact loading use one shared selection rule.
- Registry hooks expose accurate size, cache, and deletion behavior.
- At least one complete Transformers.js pipeline passes with the provider.
- If you build on an experimental contract, your package pins the Transformers.js version it was tested with.
