import { Tensor } from './utils/tensor.js';

/**
 * A cache class that stores past key values as named tensors.
 * Tensors are stored as own enumerable properties, so spread (`...cache`),
 * bracket access (`cache[name]`), and `Object.assign` work naturally.
 */
class _DynamicCache {
    /**
     * Create a DynamicCache, optionally pre-populated with entries.
     * @param {Record<string, Tensor>} [entries] Initial name→Tensor mappings.
     */
    constructor(entries) {
        if (entries) {
            Object.assign(this, entries);
        }
    }

    /**
     * Get the cached sequence length.
     * For hybrid models (e.g., Qwen3.5), finds a standard `past_key_values.*`
     * entry to determine the true past length.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);
        for (const name in self) {
            if (name.startsWith('past_key_values.')) {
                return self[name].dims.at(-2);
            }
        }
        throw new Error('Unable to determine sequence length from the cache.');
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
 * @typedef {_DynamicCache & Record<string, Tensor>} DynamicCache
 */

export const DynamicCache = /** @type {new (entries?: Record<string, Tensor>) => DynamicCache} */ (
    /** @type {unknown} */ (_DynamicCache)
);
