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

import { installGenerationRuntime } from '../generation/runtime.js';

/**
 * @typedef {Object} InferenceModel
 * @property {(inputs: Record<string, import('../utils/tensor.js').Tensor>) => Promise<Record<string, import('../utils/tensor.js').Tensor>>} [forward]
 * @property {(options: Object) => Promise<import('../utils/tensor.js').Tensor|Object>} [generate]
 * @property {import('../generation/runtime.js').GenerationCapabilitiesV1} [generationCapabilities]
 * @property {(options: Object) => Promise<import('../generation/runtime.js').AutoregressiveSessionV1>} [createAutoregressiveSession]
 * @property {Object} [config]
 * @property {() => Promise<unknown>|unknown} dispose
 */

/**
 * @typedef {Object} InferenceBackend
 * @property {string} modelId Model ID or local path used for shared config, tokenizer, and processor assets.
 * @property {(options: any) => Promise<InferenceModel|Function>} load
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
 * @param {Object} options
 * @returns {Promise<InferenceModel|Function>}
 */
export async function loadInferenceModel(backend, options) {
    const loadOptions = { ...options, modelId: backend.modelId };
    if (loadOptions.device === null) loadOptions.device = undefined;
    if (loadOptions.dtype === null) loadOptions.dtype = undefined;
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
