/// <reference types="@webgpu/types" />
import { Tensor } from './utils/tensor.js';
import { createGpuBufferTensor, isONNXProxy } from './backends/onnx.js';

/**
 * Map a cache output name to the corresponding cache input name.
 * @param {string} name The output name.
 * @returns {string} The matching input name.
 */
export function presentNameToPastName(name) {
    return (
        name
            // Hybrid cache architecture
            .replace('present_ssm', 'past_ssm') // Mamba
            .replace('present_conv', 'past_conv') // LFM2
            .replace('present_recurrent', 'past_recurrent') // Qwen3.5
            .replace('present_compressor', 'past_compressor') // Deepseek V4
            .replace('present_indexer', 'past_indexer') // Deepseek V4

            // Standard cache architecture
            .replace('present', 'past_key_values')
    );
}

/**
 * A cache class that stores past key values as named tensors.
 */
class _DynamicCache {
    /**
     * Create a DynamicCache, optionally pre-populated with entries.
     * @param {Record<string, Tensor>} [entries] Initial name→Tensor mappings.
     */
    constructor(entries) {
        if (!entries) return;
        for (const key in entries) {
            if (key in this) {
                throw new TypeError(`Key "${key}" conflicts with an existing property on DynamicCache`);
            }
            const value = entries[key];
            if (!(value instanceof Tensor)) {
                throw new TypeError(`Expected a Tensor for key "${key}", got ${typeof value}`);
            }
            this[key] = value;
        }
    }

    /**
     * Get the cached sequence length. This requires at least one attention cache entry to be present.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);

        if (Object.keys(self).length === 0) {
            return 0;
        }

        for (const name in self) {
            if (name.startsWith('past_key_values.')) {
                return self[name].dims.at(-2);
            }
        }
        throw new Error('Unable to determine sequence length from the cache.');
    }

    /**
     * Update the cache in-place with new entries, disposing replaced GPU tensors.
     * @param {Record<string, Tensor>} newEntries The new name → Tensor mappings.
     */
    update(newEntries) {
        for (const key in newEntries) {
            const oldValue = this[key];
            const newValue = newEntries[key];
            if (oldValue && oldValue !== newValue && oldValue.location === 'gpu-buffer') {
                oldValue.dispose();
            }
            this[key] = newValue;
        }
    }

    /**
     * Dispose all contained tensors whose data resides on the GPU.
     * Returns a promise that resolves when all disposals are complete.
     * @returns {Promise<void>} Promise that resolves when all GPU tensors are disposed.
     */
    async dispose() {
        const promises = [];
        for (const t of /** @type {Tensor[]} */ (Object.values(this))) {
            if (t.location === 'gpu-buffer') {
                promises.push(t.dispose());
            }
        }
        await Promise.all(promises);
    }
}

/**
 * @typedef {Record<string, Tensor> & _DynamicCache} DynamicCache
 */

export const DynamicCache = /** @type {new (entries?: Record<string, Tensor>) => DynamicCache} */ (
    /** @type {unknown} */ (_DynamicCache)
);

/**
 * A cache with a fixed, pre-allocated capacity, held entirely in GPU buffers.
 *
 * Each entry is allocated once at `max_cache_len` and the model's cache outputs
 * are written in place onto the same buffers via run-fetches.
 *
 * Constraints: WebGPU only, decoder-only models, batch size 1, and the exported
 * graph must support past/present sharing one buffer (`past_present_share_buffer`
 * semantics, true for GQA-based exports).
 *
 * The cache is owned by the caller and must be disposed:
 * ```js
 * const past_key_values = new StaticCache({ max_cache_len: 4096 });
 * await generator(messages, { max_new_tokens: 256, past_key_values });
 * await past_key_values.dispose();
 * ```
 */
class _StaticCache {
    /** @type {number} */
    #max_cache_len;
    /** @type {number} */
    #seq_length = 0;
    /** @type {Record<string, Tensor|null>|null} Output name -> in-place output tensor (or null = ORT-allocated). */
    #fetches = null;
    /** @type {GPUBuffer[]} GPU buffers owned by this cache. */
    #buffers = [];
    /** @type {boolean} */
    #disposed = false;
    /** @type {boolean} */
    #in_use = false;

    /**
     * @param {Object} options
     * @param {number} options.max_cache_len Maximum total sequence length (prompt + generated tokens).
     */
    constructor({ max_cache_len }) {
        if (!Number.isInteger(max_cache_len) || max_cache_len <= 0) {
            throw new Error(`StaticCache requires a positive integer \`max_cache_len\`, got ${max_cache_len}.`);
        }
        this.#max_cache_len = max_cache_len;
    }

    get max_cache_len() {
        return this.#max_cache_len;
    }

    get allocated() {
        return this.#fetches !== null;
    }

    #assertNotDisposed() {
        if (this.#disposed) {
            throw new Error('StaticCache has been disposed. Create a new StaticCache instead.');
        }
    }

    /**
     * Get the cached sequence length.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        return this.#seq_length;
    }

    /**
     * No-op: cache outputs are written in place, so there is nothing to store.
     * @param {Record<string, Tensor>} newEntries Ignored.
     */
    update(newEntries) {}

    /**
     * Allocate the cache buffers from the decoder session's metadata.
     *
     * Allocation is transactional: entries are first built into local structures and only
     * committed onto the cache once every allocation has succeeded.
     *
     * @param {Object} session The decoder InferenceSession (with `.config.device` attached).
     * @param {Set<string>} cacheInputNames Cache input names (from `getCacheNames`).
     * @param {Record<string, number>} symbols Resolved symbolic dims (e.g. `{ batch_size: 1 }`).
     * @returns {Promise<void>}
     * @internal
     */
    async _allocate(session, cacheInputNames, symbols) {
        this.#assertNotDisposed();
        if (session.config?.device !== 'webgpu') {
            throw new Error(
                `StaticCache is only supported on the 'webgpu' device (got '${session.config?.device}'). ` +
                    'Use DynamicCache (the default) instead.',
            );
        }
        if (isONNXProxy()) {
            throw new Error('StaticCache is not supported when the ONNX backend is proxied.');
        }

        /** @type {Record<string, Tensor>} */
        const entries = Object.create(null);
        /** @type {GPUBuffer[]} */
        const buffers = [];
        try {
            for (const meta of session.inputMetadata) {
                if (!cacheInputNames.has(meta.name)) continue;

                const isAttention = meta.name.startsWith('past_key_values.');
                const dims = meta.shape.map((d, i) => {
                    if (typeof d === 'number' && d > 0) return d;
                    if (symbols[d] !== undefined) return symbols[d];
                    if (isAttention && i === meta.shape.length - 2) return this.#max_cache_len;
                    throw new Error(
                        `StaticCache: unable to resolve dimension ${i} (${JSON.stringify(d)}) of cache input ` +
                            `'${meta.name}' (shape: [${meta.shape.map((x) => JSON.stringify(x)).join(', ')}]). ` +
                            'Use DynamicCache (the default) instead.',
                    );
                });
                const { tensor, gpuBuffer } = await createGpuBufferTensor(meta.type, dims);
                // Track the buffer before wrapping so it is destroyed on a later failure.
                buffers.push(gpuBuffer);
                entries[meta.name] = new Tensor(tensor);
            }
            if (buffers.length === 0) {
                throw new Error(
                    'StaticCache: the model session has no cache inputs. ' +
                        'Use DynamicCache (the default) for models without a KV cache.',
                );
            }

            const fetches = Object.create(null);
            for (const meta of session.outputMetadata) {
                const name = meta.name;
                const past = presentNameToPastName(name);
                if (name === past || !(entries[past] instanceof Tensor)) {
                    fetches[name] = null;
                    continue;
                }
                this.#checkShareBufferSemantics(name, meta.shape, past, entries[past].dims);
                fetches[name] = entries[past];
            }

            // Only reached once every allocation has succeeded.
            Object.assign(this, entries);
            this.#buffers = buffers;
            this.#fetches = fetches;
        } catch (e) {
            // Release the ORT tensor wrappers, then destroy the underlying GPU buffers.
            for (const tensor of Object.values(entries)) {
                tensor.dispose();
            }
            for (const buffer of buffers) {
                if (typeof buffer.destroy === 'function') buffer.destroy();
            }
            throw e;
        }
    }

    /**
     * Check whether a `present.*` output can safely share the `past.*` buffer.
     * Rejects concatenation-style shapes whose sequence axis grows with the past length
     * as they are incompatible with fixed-size buffers.
     *
     * @param {string} presentName The output name.
     * @param {ReadonlyArray<number|string>} presentShape The output's metadata shape.
     * @param {string} pastName The matching cache input name.
     * @param {ReadonlyArray<number>} allocatedDims The dims the cache entry was allocated with.
     */
    #checkShareBufferSemantics(presentName, presentShape, pastName, allocatedDims) {
        if (!pastName.startsWith('past_key_values.')) return;

        const axis = allocatedDims.length - 2;
        const presentDim = presentShape?.[axis];
        if (typeof presentDim === 'string' && presentDim.includes('+')) {
            throw new Error(
                `StaticCache requires past/present buffer sharing (\`past_present_share_buffer\`), but ` +
                    `'${presentName}' declares its sequence axis as '${presentDim}', which grows with the past length. ` +
                    'Use DynamicCache (the default) instead.',
            );
        }
        if (typeof presentDim === 'number' && presentDim !== allocatedDims[axis]) {
            throw new Error(
                `StaticCache: '${presentName}' has a fixed sequence length of ${presentDim}, ` +
                    `which does not match the allocated cache length (${allocatedDims[axis]}). ` +
                    `Set \`max_cache_len\` to ${presentDim} or use DynamicCache (the default) instead.`,
            );
        }
    }

    /**
     * Mark the cache as in use by a generation.
     * @internal
     */
    _acquire() {
        this.#assertNotDisposed();
        if (this.#in_use) {
            throw new Error(
                'This StaticCache is already in use by a concurrent generation. ' +
                    'A StaticCache can only serve one generation at a time.',
            );
        }
        this.#in_use = true;
    }

    /**
     * Release the cache after a generation has finished.
     * @internal
     */
    _release() {
        this.#in_use = false;
    }

    /**
     * @internal
     */
    _getFetches() {
        return this.#fetches;
    }

    /**
     * Assert that the run wrote every cache output in place. A substituted output would leave the new
     * KV data outside the cache's buffers, so fail loudly before the sequence length is committed.
     * @param {Record<string, Tensor>} outputs The outputs returned by the run.
     * @internal
     */
    _assertWrittenInPlace(outputs) {
        if (!this.#fetches) return;
        for (const [name, tensor] of Object.entries(this.#fetches)) {
            if (tensor !== null && outputs[name] !== tensor) {
                throw new Error(
                    `StaticCache: the '${name}' output was not written into the pre-allocated cache tensor. ` +
                        'Use DynamicCache (the default) instead.',
                );
            }
        }
    }

    /**
     * Check that `numTokens` more tokens fit in the cache, without changing its state.
     * @param {number} numTokens Number of tokens about to be written.
     * @internal
     */
    _checkCapacity(numTokens) {
        this.#assertNotDisposed();
        if (this.#seq_length + numTokens > this.#max_cache_len) {
            throw new Error(
                `StaticCache capacity exceeded: ${this.#seq_length} cached + ${numTokens} new tokens > ` +
                    `max_cache_len (${this.#max_cache_len}). Increase \`max_cache_len\`.`,
            );
        }
    }

    /**
     * Advance the cached sequence length. Call only after the run has succeeded.
     * @param {number} numTokens Number of tokens that were written.
     * @internal
     */
    _commit(numTokens) {
        this.#seq_length += numTokens;
    }

    /**
     * Release the cache's tensors and destroy its GPU buffers. Idempotent; any
     * further use of the cache throws.
     * @returns {Promise<void>}
     */
    async dispose() {
        if (this.#disposed) return;
        if (this.#in_use) {
            throw new Error('Cannot dispose a StaticCache while it is in use by a generation.');
        }
        this.#disposed = true;
        // Release the wrappers (drops references only); the buffers are destroyed below.
        for (const key of Object.keys(this)) {
            const tensor = this[key];
            if (tensor instanceof Tensor) {
                tensor.dispose();
            }
            delete this[key];
        }
        for (const buffer of this.#buffers) {
            if (typeof buffer.destroy === 'function') buffer.destroy();
        }
        this.#buffers = [];
        this.#fetches = null;
        this.#seq_length = 0;
    }
}

/**
 * @typedef {Record<string, Tensor> & _StaticCache} StaticCache
 */

export const StaticCache = /** @type {new (options: { max_cache_len: number }) => StaticCache} */ (
    /** @type {unknown} */ (_StaticCache)
);
