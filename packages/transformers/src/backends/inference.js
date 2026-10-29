/**
 * @file Runtime-neutral inference backend helpers.
 *
 * A generic inference backend is a model factory with a shared pretrained model ID:
 *
 * ```js
 * const backend = {
 *     modelId: 'organization/model',
 *     async load(options) {
 *         return model;
 *     },
 * };
 * ```
 *
 * The returned model may be callable, or may expose a `forward(inputs)` method.
 * Generation models expose `createAutoregressiveSession(options)`; Transformers.js installs their public `generate()`.
 *
 * Contract stability. The following parts are stable: `modelId`, `load()`, static `capabilities`,
 * `sharedAssets`, `chatTemplate`, the registry hooks, the host services passed to `load()`, loaded
 * models with `forward()` and `dispose()`, and the `forward` capability. Everything marked
 * "Experimental" below is being developed together with a first external runtime and may change
 * in a minor release: the normalized-session contract (including `getSessionConfig()`), causal
 * generation, provider-owned `generate()`, model-owned pipeline steps, and the random-access
 * `artifactProvider`.
 *
 * @module backends/inference
 */

import { getCausalGenerationCapabilities, installGenerationRuntime } from '../generation/runtime.js';
import { validateInferenceArtifactProvider } from './artifacts.js';
import { env } from '../env.js';
import { getModelFile } from '../utils/hub.js';
import { get_file_metadata } from '../utils/model_registry/get_file_metadata.js';
import { delete_file_from_cache } from '../utils/model_registry/delete_file_from_cache.js';

/**
 * @typedef {Object} ForwardCapabilitiesV1
 * @property {1} version
 */

/**
 * Execution capabilities of a loaded inference model. Every listed capability has a consumer in
 * Transformers.js. New capability families are added here together with the code that reads them.
 *
 * @typedef {Object} InferenceModelCapabilities
 * @property {ForwardCapabilitiesV1} [forward] Stable. The model implements `forward(inputs)`.
 * @property {import('../generation/runtime.js').CausalGenerationCapabilitiesV1} [causalGeneration] Experimental: The model implements `createAutoregressiveSession(options)`.
 * @property {PipelineStepsCapabilitiesV1} [pipelineSteps] Experimental: The model owns preprocessing and/or postprocessing for the listed tasks.
 * @property {{version: 1, preprocess?: 'model', postprocess?: 'model'}} [objectDetection] Deprecated alias for `pipelineSteps` with the `object-detection` task.
 */

/**
 * Tasks whose pipeline steps a loaded model can take over, per step. Each entry has a fixed,
 * task-specific contract documented in the integration guide.
 */
export const MODEL_PIPELINE_STEP_TASKS = Object.freeze({
    preprocess: Object.freeze(['image-classification', 'object-detection']),
    postprocess: Object.freeze(['object-detection']),
});

/**
 * Experimental: declares which pipeline steps a loaded model performs itself. A task listed under
 * `preprocess` makes the pipeline call `model.preprocess(task, inputs)` instead of its processor and
 * pass the result to the model. A task listed under `postprocess` makes the pipeline call
 * `model.postprocess(task, outputs, options)` instead of the processor's postprocessing.
 *
 * @typedef {Object} PipelineStepsCapabilitiesV1
 * @property {1} version
 * @property {ReadonlyArray<string>} [preprocess] Tasks from `MODEL_PIPELINE_STEP_TASKS.preprocess`.
 * @property {ReadonlyArray<string>} [postprocess] Tasks from `MODEL_PIPELINE_STEP_TASKS.postprocess`.
 */

/**
 * Advisory capabilities available before a curated backend is loaded.
 * `load()` remains authoritative for device and dtype validation.
 *
 * @typedef {Object} StaticBackendCapabilities
 * @property {ReadonlyArray<string>} devices
 * @property {ReadonlyArray<string>} dtypes
 * @property {ReadonlyArray<string>} tasks
 */

/**
 * Versioned capability for providers that construct the normalized sessions consumed by built-in
 * Transformers.js model classes.
 *
 * Experimental: the session roles and file names passed to `constructSessions()` come from the
 * provider's `getSessionConfig()`, or from the built-in ONNX session configuration when the
 * provider does not define it. A portable execution manifest is planned to replace them.
 *
 * @typedef {Object} SessionProviderCapabilitiesV1
 * @property {1} version
 */

/**
 * Runtime-neutral session consumed by built-in Transformers.js model classes.
 *
 * Experimental: see {@link SessionProviderCapabilitiesV1}.
 *
 * @typedef {Object} InferenceSession
 * @property {ReadonlyArray<string>} inputNames
 * @property {ReadonlyArray<string>} outputNames
 * @property {unknown} [inputMetadata]
 * @property {unknown} [outputMetadata]
 * @property {Readonly<Record<string, unknown>>} [config]
 * @property {(inputs: Record<string, import('../utils/tensor.js').Tensor>) => Promise<Record<string, import('../utils/tensor.js').Tensor>>} run
 * @property {() => Promise<unknown>|unknown} release
 */

/**
 * Session roles for one model architecture, returned by `getSessionConfig()`.
 *
 * Experimental: see {@link SessionProviderCapabilitiesV1}.
 *
 * @typedef {Object} InferenceSessionConfig
 * @property {Record<string, string>} sessions Session role to base file name.
 * @property {Record<string, boolean>} [cache_sessions] Session roles that produce key-value cache (`present.*`) outputs.
 * @property {Record<string, string>} [optional_configs] Additional config files to load, keyed by name.
 */

/**
 * @typedef {import('../utils/hub.js').PretrainedModelOptions & {
 *   modelId: string,
 *   fetch: typeof globalThis.fetch,
 *   getModelFile: typeof getInferenceBackendModelFile,
 *   getModelFileMetadata: typeof getInferenceBackendModelFileMetadata,
 *   deleteModelFile: typeof deleteInferenceBackendModelFile,
 *   task?: string,
 *   config?: import('../configs.js').PretrainedConfig,
 *   modelClass?: Function,
 *   generation_config?: Record<string, unknown>,
 *   artifactMetadata?: Record<string, {size?: number, fromCache?: boolean}>,
 * }} InferenceBackendLoadOptions
 */

/**
 * @typedef {import('../utils/hub.js').PretrainedModelOptions & {
 *   fetch?: typeof globalThis.fetch,
 *   task?: string,
 *   config?: import('../configs.js').PretrainedConfig,
 *   modelClass?: Function,
 *   generation_config?: Record<string, unknown>,
 *   artifactMetadata?: Record<string, {size?: number, fromCache?: boolean}>,
 * }} InferenceModelLoadOptions
 */

/**
 * @typedef {Object} InferenceModel
 * @property {(inputs: Record<string, import('../utils/tensor.js').Tensor>) => Promise<Record<string, import('../utils/tensor.js').Tensor>>} [forward]
 * @property {(options: Record<string, unknown>) => Promise<import('../utils/tensor.js').Tensor|Record<string, unknown>>} [generate] Experimental: Provider-owned generation, called by generating pipelines such as automatic speech recognition and text2text generation. A model may implement `generate()` without `forward()`. When the model implements `createAutoregressiveSession()`, Transformers.js installs its own `generate()` and replaces this one.
 * @property {InferenceModelCapabilities} [capabilities]
 * @property {(options: import('../generation/runtime.js').AutoregressiveSessionOptionsV1) => Promise<import('../generation/runtime.js').AutoregressiveSessionV1>} [createAutoregressiveSession] Experimental: Required with `capabilities.causalGeneration`.
 * @property {(task: string, inputs: unknown, options?: Record<string, unknown>) => Promise<Record<string, unknown>>|Record<string, unknown>} [preprocess] Experimental: Required with `capabilities.pipelineSteps.preprocess`. Returns the model inputs.
 * @property {(task: string, outputs: Record<string, import('../utils/tensor.js').Tensor>, options?: Record<string, unknown>) => Promise<unknown>|unknown} [postprocess] Experimental: Required with `capabilities.pipelineSteps.postprocess`.
 * @property {(images: import('../utils/image.js').RawImage[]) => Promise<Record<string, unknown>>} [preprocessObjectDetection] Deprecated: use `preprocess()`. Used when `capabilities.objectDetection.preprocess === 'model'`.
 * @property {(outputs: Record<string, import('../utils/tensor.js').Tensor>, threshold?: number, targetSizes?: number[][]|null) => Promise<unknown>|unknown} [postProcessObjectDetection] Deprecated: use `postprocess()`. Used when `capabilities.objectDetection.postprocess === 'model'`.
 * @property {import('../configs.js').PretrainedConfig} [config]
 * @property {() => Promise<unknown>|unknown} dispose
 */

/**
 * A backend-provided default chat template. File sources default to the backend model ID and
 * `chat_template.jinja`; inline content is never fetched or cached.
 *
 * @typedef {
 *   | {content: string, modelId?: never, file?: never}
 *   | {content?: never, modelId?: string, revision?: string, subfolder?: string, file?: string}
 * } InferenceBackendChatTemplate
 */

/**
 * @typedef {Object} InferenceBackend
 * @property {string} modelId Model ID or local path used for shared config, tokenizer, and processor assets.
 * @property {{revision?: string, subfolder?: string}} [sharedAssets] Backend-selected location details for shared Transformers.js assets.
 * @property {InferenceBackendChatTemplate} [chatTemplate] Default chat template installed on a pipeline tokenizer.
 * @property {StaticBackendCapabilities} [capabilities]
 * @property {SessionProviderCapabilitiesV1} [sessionProvider] Experimental: Declares support for normalized built-in model sessions.
 * @property {(names: Record<string, string>, options: InferenceBackendLoadOptions, cacheSessions?: Record<string, boolean>) => Promise<Record<string, InferenceSession>>} [constructSessions] Experimental: Required with `sessionProvider`.
 * @property {(modelType: string, config: Object, options: Object & {textOnly: boolean}) => InferenceSessionConfig|null} [getSessionConfig] Experimental: Maps an architecture to session roles and base file names passed to `constructSessions()`. Returns `null` when the provider cannot run the architecture. Defaults to the built-in session configuration.
 * @property {(options: Object) => ReadonlyArray<string>|Promise<ReadonlyArray<string>>} [listModelArtifacts] Lists backend-owned files required for the selected load options.
 * @property {(file: string, options: Object) => Promise<{size?: number, fromCache?: boolean}|null>} [getModelArtifactMetadata] Returns backend-owned cache metadata for an artifact.
 * @property {(file: string, options: Object) => Promise<boolean>} [deleteModelArtifact] Deletes an artifact from backend-owned cache storage.
 * @property {(options: InferenceBackendLoadOptions) => Promise<InferenceModel|Function>} [load] Required unless `sessionProvider` and `constructSessions()` are implemented.
 */

/**
 * Returns whether a value implements either the generic model-factory contract or the versioned
 * normalized-session contract. Classes with static members are supported too.
 *
 * @param {unknown} value
 * @returns {value is InferenceBackend}
 */
export function isInferenceBackend(value) {
    const backend = /** @type {any} */ (value);
    return (
        (typeof value === 'object' || typeof value === 'function') &&
        value !== null &&
        typeof backend.modelId === 'string' &&
        (typeof backend.load === 'function' ||
            (backend.sessionProvider?.version === 1 && typeof backend.constructSessions === 'function'))
    );
}

/**
 * Returns whether a backend implements the versioned normalized-session contract used by built-in
 * Transformers.js model classes.
 *
 * @param {unknown} value
 * @returns {value is InferenceBackend & {sessionProvider: SessionProviderCapabilitiesV1, constructSessions: Function}}
 */
export function isSessionInferenceProvider(value) {
    const provider = /** @type {any} */ (value);
    return (
        (typeof value === 'object' || typeof value === 'function') &&
        value !== null &&
        typeof provider.modelId === 'string' &&
        provider.sessionProvider?.version === 1 &&
        typeof provider.constructSessions === 'function'
    );
}

/**
 * Returns whether a backend is the built-in ONNX session provider. Session providers load
 * Transformers.js model classes rather than returning runtime-neutral inference models.
 *
 * @param {unknown} value
 * @returns {boolean}
 */
export function isOnnxSessionProvider(value) {
    const provider = /** @type {any} */ (value);
    return isSessionInferenceProvider(value) && provider.providerType === 'onnx';
}

/**
 * Resolve a string model ID from either a string or an inference backend.
 *
 * @param {string|InferenceBackend} model
 * @returns {string}
 */
export function getModelId(model) {
    if (typeof model === 'string') return model;
    if (isInferenceBackend(model)) return model.modelId;
    throw new TypeError(
        'Model must be a model ID string or an inference backend with `modelId` and either `load(options)` or the normalized-session contract.',
    );
}

/**
 * @typedef {Object & {
 *   fatal?: boolean,
 *   returnPath?: boolean,
 * }} InferenceBackendModelFileOptions Pretrained loading options plus two artifact-specific flags.
 * `fatal` (default `true`) rejects on a missing file instead of resolving `null`. `returnPath`
 * (default `false`) returns a filesystem path instead of bytes where the environment supports it.
 */

/**
 * Load a model artifact through Transformers.js transport, progress, and cache handling.
 *
 * @param {string} modelId
 * @param {string} file
 * @param {InferenceBackendModelFileOptions} [options]
 * @returns {Promise<string|Uint8Array|null>}
 */
export async function getInferenceBackendModelFile(modelId, file, options = {}) {
    const { fatal = true, returnPath = false, ...loadOptions } = options;
    return getModelFile(modelId, file, fatal, loadOptions, returnPath);
}

/**
 * Read model artifact metadata through Transformers.js cache-aware discovery.
 *
 * @param {string} modelId
 * @param {string} file
 * @param {Object} [options]
 */
export async function getInferenceBackendModelFileMetadata(modelId, file, options = {}) {
    return get_file_metadata(modelId, file, options);
}

/**
 * Delete one model artifact from Transformers.js cache storage.
 *
 * @param {string} modelId
 * @param {string} file
 * @param {Object} [options]
 * @returns {Promise<boolean>}
 */
export async function deleteInferenceBackendModelFile(modelId, file, options = {}) {
    return delete_file_from_cache(modelId, file, options);
}

/**
 * Add host-owned services to options passed across the inference-backend boundary.
 * The host fetch implementation is authoritative so every backend request observes
 * Transformers.js environment customization.
 *
 * @param {Object} [options]
 * @returns {Object & {
 *   fetch: typeof globalThis.fetch,
 *   getModelFile: typeof getInferenceBackendModelFile,
 *   getModelFileMetadata: typeof getInferenceBackendModelFileMetadata,
 *   deleteModelFile: typeof deleteInferenceBackendModelFile,
 * }}
 */
export function withInferenceBackendHostOptions(options = {}) {
    return {
        ...options,
        fetch: env.fetch,
        getModelFile: getInferenceBackendModelFile,
        getModelFileMetadata: getInferenceBackendModelFileMetadata,
        deleteModelFile: deleteInferenceBackendModelFile,
    };
}

/**
 * Apply backend-selected shared-asset location details. Source-specific values
 * take precedence over per-load options, which retain their normal defaults.
 *
 * @param {InferenceBackend} backend
 * @param {Object} [options]
 * @returns {Object}
 */
export function withInferenceBackendSharedAssetOptions(backend, options = {}) {
    const source = backend.sharedAssets;
    if (!source) return { ...options };
    return {
        ...options,
        revision: source.revision ?? options.revision,
        subfolder: source.subfolder ?? options.subfolder,
    };
}

/**
 * Reject a task excluded by authoritative static backend metadata.
 *
 * @param {InferenceBackend} backend
 * @param {string} task
 */
export function validateInferenceBackendTask(backend, task) {
    const tasks = backend.capabilities?.tasks;
    if (!tasks) return;
    const canonicalTask = task.split('_', 1)[0];
    if (!tasks.includes(task) && !tasks.includes(canonicalTask)) {
        throw new Error(`Inference backend "${backend.modelId}" does not support the "${task}" task.`);
    }
}

/**
 * Validate the loaded execution capability required by an integrated task.
 *
 * @param {InferenceModel|Function} model
 * @param {string} task
 */
export function validateInferenceModelTask(model, task) {
    const implementation = /** @type {any} */ (model);
    if (!implementation.capabilities) return;
    if (task === 'text-generation') {
        if (
            !getCausalGenerationCapabilities(implementation) ||
            typeof implementation.createAutoregressiveSession !== 'function'
        ) {
            throw new Error('The loaded inference model does not support causal text generation.');
        }
    }
}

/**
 * Validate a loaded model's `capabilities.pipelineSteps` declaration.
 *
 * @param {any} implementation
 */
function validatePipelineSteps(implementation) {
    const steps = implementation.capabilities?.pipelineSteps;
    if (steps == null) return;
    if (steps.version !== 1) {
        throw new TypeError('`capabilities.pipelineSteps` must declare `version: 1`.');
    }
    for (const step of /** @type {const} */ (['preprocess', 'postprocess'])) {
        const tasks = steps[step];
        if (tasks == null) continue;
        if (!Array.isArray(tasks)) {
            throw new TypeError(`\`capabilities.pipelineSteps.${step}\` must be an array of task names.`);
        }
        const unsupported = tasks.filter((task) => !MODEL_PIPELINE_STEP_TASKS[step].includes(task));
        if (unsupported.length > 0) {
            throw new TypeError(
                `\`capabilities.pipelineSteps.${step}\` lists unsupported task(s): ${unsupported.join(', ')}. ` +
                    `Supported tasks: ${MODEL_PIPELINE_STEP_TASKS[step].join(', ')}.`,
            );
        }
        if (tasks.length > 0 && typeof implementation[step] !== 'function') {
            throw new TypeError(
                `Models declaring \`capabilities.pipelineSteps.${step}\` must implement \`${step}(task, ...)\`.`,
            );
        }
    }
}

/**
 * Return the model-owned implementation of a pipeline step for a task, or `null` when the pipeline
 * should run its own step. The returned function takes the step's arguments without the task.
 *
 * @param {unknown} model
 * @param {'preprocess'|'postprocess'} step
 * @param {string} task
 * @returns {((...args: any[]) => Promise<any>)|null}
 */
export function getModelPipelineStep(model, step, task) {
    const implementation = /** @type {any} */ (model);
    const steps = implementation?.capabilities?.pipelineSteps;
    if (steps?.version === 1 && steps[step]?.includes(task) && typeof implementation[step] === 'function') {
        return async (...args) => implementation[step](task, ...args);
    }

    // Deprecated `capabilities.objectDetection` alias.
    const legacy = implementation?.capabilities?.objectDetection;
    if (task === 'object-detection' && legacy?.version === 1 && legacy[step] === 'model') {
        if (step === 'preprocess' && typeof implementation.preprocessObjectDetection === 'function') {
            return async (images) => implementation.preprocessObjectDetection(images);
        }
        if (step === 'postprocess' && typeof implementation.postProcessObjectDetection === 'function') {
            return async (outputs, { threshold, targetSizes } = {}) =>
                implementation.postProcessObjectDetection(outputs, threshold, targetSizes);
        }
    }
    return null;
}

/**
 * Make a plain model with `forward()` callable, matching the model contract used by pipelines.
 *
 * @param {InferenceModel|Function} model
 * @returns {InferenceModel|Function}
 */
export function normalizeInferenceModel(model) {
    const implementation = /** @type {any} */ (model);
    if ((typeof model !== 'object' && typeof model !== 'function') || model === null) {
        throw new TypeError('Inference backend `load()` must return a model.');
    }
    if (typeof implementation.dispose !== 'function') {
        throw new TypeError('Inference backend models must implement `dispose()`.');
    }
    if (
        implementation.capabilities?.causalGeneration &&
        typeof implementation.createAutoregressiveSession !== 'function'
    ) {
        throw new TypeError(
            'Models declaring `capabilities.causalGeneration` must implement `createAutoregressiveSession(options)`.',
        );
    }
    validatePipelineSteps(implementation);
    if (typeof model === 'function') return model;
    if (
        typeof implementation.forward !== 'function' &&
        typeof implementation.generate !== 'function' &&
        typeof implementation.createAutoregressiveSession !== 'function'
    ) {
        throw new TypeError(
            'Inference backend models must be callable, implement `forward(inputs)`, implement `generate(options)`, or implement `createAutoregressiveSession(options)`.',
        );
    }

    const callable = (...args) => {
        if (typeof implementation.forward !== 'function') {
            throw new Error('This inference model does not implement `forward(inputs)`.');
        }
        return implementation.forward(...args);
    };
    return new Proxy(callable, {
        get(target, property, receiver) {
            return property in implementation
                ? Reflect.get(implementation, property, implementation)
                : Reflect.get(target, property, receiver);
        },
        set(_target, property, value) {
            return Reflect.set(implementation, property, value, implementation);
        },
        has(target, property) {
            return property in implementation || property in target;
        },
    });
}

/**
 * Load and normalize a custom inference model.
 *
 * @param {InferenceBackend} backend
 * @param {InferenceModelLoadOptions} options
 * @returns {Promise<InferenceModel|Function>}
 */
export async function loadInferenceModel(backend, options) {
    const loadOptions = withInferenceBackendHostOptions({ ...options, modelId: backend.modelId });
    if (loadOptions.device === null) loadOptions.device = undefined;
    if (loadOptions.dtype === null) loadOptions.dtype = undefined;
    validateInferenceArtifactProvider(loadOptions.artifactProvider);
    const model = /** @type {any} */ (
        installGenerationRuntime(normalizeInferenceModel(await backend.load(loadOptions)))
    );
    if (model.config == null && options.config != null) {
        model.config = options.config;
    }
    if (model.generation_config == null && options.generation_config != null) {
        model.generation_config = options.generation_config;
    }
    return model;
}
