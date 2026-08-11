/**
 * @file Runtime-neutral inference backend helpers.
 *
 * An inference backend is a model factory with a shared pretrained model ID:
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
 * @module backends/inference
 */

import { getCausalGenerationCapabilities, installGenerationRuntime } from '../generation/runtime.js';
import { validateInferenceArtifactProvider } from './artifacts.js';

/**
 * @typedef {Object} ForwardCapabilitiesV1
 * @property {1} version
 */

/**
 * Execution capabilities of a loaded inference model. Capability families without a versioned
 * Transformers.js integration remain opaque until their task-specific session contract is defined.
 *
 * @typedef {Object} InferenceModelCapabilities
 * @property {ForwardCapabilitiesV1} [forward]
 * @property {import('../generation/runtime.js').CausalGenerationCapabilitiesV1} [causalGeneration]
 * @property {Readonly<Record<string, unknown>>} [encoderDecoderGeneration]
 * @property {Readonly<Record<string, unknown>>} [diffusion]
 * @property {Readonly<Record<string, unknown>>} [audioGeneration]
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
 * @typedef {import('../utils/hub.js').PretrainedModelOptions & {
 *   modelId: string,
 *   task?: string,
 *   config?: import('../configs.js').PretrainedConfig,
 *   modelClass?: Function,
 *   generation_config?: Record<string, unknown>,
 *   artifactMetadata?: Record<string, {size?: number, fromCache?: boolean}>,
 * }} InferenceBackendLoadOptions
 */

/**
 * @typedef {import('../utils/hub.js').PretrainedModelOptions & {
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
 * @property {(options: Record<string, unknown>) => Promise<import('../utils/tensor.js').Tensor|Record<string, unknown>>} [generate]
 * @property {InferenceModelCapabilities} [capabilities]
 * @property {import('../generation/runtime.js').GenerationCapabilitiesV1} [generationCapabilities] Deprecated flat causal-generation capabilities.
 * @property {(options: import('../generation/runtime.js').AutoregressiveSessionOptionsV1) => Promise<import('../generation/runtime.js').AutoregressiveSessionV1>} [createAutoregressiveSession]
 * @property {import('../configs.js').PretrainedConfig} [config]
 * @property {() => Promise<unknown>|unknown} dispose
 */

/**
 * A backend-provided default chat template. File sources default to the backend model ID and
 * `chat_template.jinja`; inline content is never fetched or cached.
 *
 * @typedef {
 *   | {content: string, modelId?: never, file?: never}
 *   | {content?: never, modelId?: string, file?: string}
 * } InferenceBackendChatTemplate
 */

/**
 * @typedef {Object} InferenceBackend
 * @property {string} modelId Model ID or local path used for shared config, tokenizer, and processor assets.
 * @property {InferenceBackendChatTemplate} [chatTemplate] Default chat template installed on a pipeline tokenizer.
 * @property {string} [providerType] Provider family identifier used for provider-specific host initialization.
 * @property {StaticBackendCapabilities} [capabilities]
 * @property {(options: Object) => ReadonlyArray<string>|Promise<ReadonlyArray<string>>} [listModelArtifacts] Lists backend-owned files required for the selected load options.
 * @property {(file: string, options: Object) => Promise<{size?: number, fromCache?: boolean}|null>} [getModelArtifactMetadata] Returns backend-owned cache metadata for an artifact.
 * @property {(file: string, options: Object) => Promise<boolean>} [deleteModelArtifact] Deletes an artifact from backend-owned cache storage.
 * @property {(options: InferenceBackendLoadOptions) => Promise<InferenceModel|Function>} load
 * @property {(names: Record<string, string>, options: Object, cacheSessions?: Object) => Promise<Record<string, Object>>} [constructSessions]
 */

/**
 * Returns whether a value implements the custom inference backend contract.
 * Classes with static `modelId` and `load` members are supported too.
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
        typeof backend.load === 'function'
    );
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
    throw new TypeError('Model must be a model ID string or an inference backend with `modelId` and `load(options)`.');
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
    if (typeof model === 'function') return model;
    if (
        typeof implementation.forward !== 'function' &&
        typeof implementation.createAutoregressiveSession !== 'function'
    ) {
        throw new TypeError(
            'Inference backend models must be callable, implement `forward(inputs)`, or implement `createAutoregressiveSession(options)`.',
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
    const loadOptions = { ...options, modelId: backend.modelId };
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
